import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
from torch.cuda.amp import GradScaler
import evaluate
import wandb 
from transformers import BartTokenizer, AutoConfig, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

# Import utilities and datasets
from custom_bart import BartForConditionalGeneration
from utils import (
    load_checkpoint,
    set_seed,
    setup_logging,
    log_test_metrics,
    log_metrics,
    save_checkpoint,
    add_model_specific_args,
    postprocess_text,
    device 
)
from summarization_datasets import (
    QMDataset,
    SummScreenDataset,
    GovReportDataset,
    BookSumDataset,
    MensaDataset,
    UbadaaSumDataset
)

def load_model_and_tokenizer(args):
    """
    Loads the BART model and tokenizer based on command-line arguments.

    Args:
        args (argparse.Namespace): Arguments object containing model_name.

    Returns:
        tuple: (tokenizer, model, config)
    """
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    # If gradient checkpointing is enabled
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    return tokenizer, model, config

def train_model(num_training_steps, model, optimizer, config, tokenizer, scaler, lr_scheduler,
                train_dataloader, validation_dataloader, logger, args, rouge_metric, bertscore_metric):
    """
    Custom training loop for the BART model.
    Preserves the original custom gradient accumulation and backward pass logic.

    Args:
        num_training_steps (int): Total number of training steps.
        model (torch.nn.Module): The BART model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        config (object): Model configuration.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        scaler (torch.cuda.amp.GradScaler): The GradScaler for mixed precision training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        validation_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        logger (logging.Logger): The logger instance.
        args (argparse.Namespace): Command-line arguments.
        rouge_metric (evaluate.Metric): ROUGE metric object.
        bertscore_metric (evaluate.Metric): BERTScore metric object.
    """
    progress_bar = tqdm(range(num_training_steps))
    completed_steps = 0
    step = 0
    best_val_r1 = -1

    logger.info(f"Training on device: {device}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        mean_loss = 0.0

        for batch in train_dataloader:
            # Determine batch structure based on dataset (QMDataset returns 3 items, others 2)
            if args.dataset_name == "qmsum":
                input_ids, output_ids, query_ids = batch
            elif args.dataset_name == "mensa":
                scenes, output_ids = batch # scenes is a list of strings
                input_ids = None # Not a direct tensor for processing here
            else: # summscreen, govreport, booksum
                input_ids, output_ids = batch

            # Move output_ids to device and handle padding for loss calculation
            output_ids[output_ids == tokenizer.pad_token_id] = -100

            encoder = model.get_encoder()
            tensorStack = None
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id

            optimizer.zero_grad() # Zero gradients for the accumulation step

            with torch.no_grad():
                if args.dataset_name == "mensa":
                    # Special handling for scenes
                    # Each scene needs to be tokenized and passed through encoder separately.
                    for idx in range(len(scenes)):
                        # Assuming each scene is a sub-document and should be within 1024 tokens
                        tokenized_inputs = tokenizer.batch_encode_plus(
                            scenes[idx], truncation=True, padding="do_not_pad",
                            max_length=1024, return_tensors="pt"
                        )
                        # Ensure inputs are moved to device
                        input_ids_chunk = tokenized_inputs["input_ids"].to(device)
                        attention_mask_chunk = tokenized_inputs["attention_mask"].to(device)

                        out = encoder(input_ids=input_ids_chunk, attention_mask=attention_mask_chunk)
                        if tensorStack is None:
                            tensorStack = out[0].squeeze(0)[:-1, :]
                        else:
                            # Concatenate while skipping start/end tokens of intermediate chunks
                            tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[1:-1, :]), dim=0)
                    # Add the last token of the final chunk
                    tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[-2:-1, :]), dim=0)

                else: # Common handling for QMDataset, SummScreenDataset, GovReportDataset, BookSumDataset
                    # These datasets provide a single long input_ids tensor per example
                    # We chunk the input_ids into 1024-token segments for the encoder
                    chunk_size = 1024 - 2 # Account for BOS and EOS tokens
                    if args.dataset_name == "qmsum":
                        chunk_size = 1024 - 2 - query_ids.shape[1] # QMSum specific chunking

                    for i in range(0, input_ids.shape[1], chunk_size):
                        current_chunk_ids = input_ids[0][i:i + chunk_size].tolist()
                        if args.dataset_name == "qmsum":
                            # QMSum: [BOS] + query + chunk + [EOS]
                            tokenized_inputs = [bos_token_id] + query_ids[0].tolist() + current_chunk_ids + [eos_token_id]
                        else:
                            # Other datasets: [BOS] + chunk + [EOS]
                            tokenized_inputs = [bos_token_id] + current_chunk_ids + [eos_token_id]

                        tokenized_inputs = torch.tensor(tokenized_inputs, dtype=input_ids.dtype).unsqueeze(0).to(device)
                        attention_mask_chunk = torch.ones_like(tokenized_inputs).to(device) # Mask for the chunk

                        out = encoder(input_ids=tokenized_inputs, attention_mask=attention_mask_chunk)

                        if tensorStack is None:
                            # For the first chunk, include all except the last token (which will be start of next chunk's sequence)
                            tensorStack = out[0].squeeze(0)[:-1, :]
                        else:
                            # For intermediate chunks, include all except first and last tokens
                            tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[1:-1, :]), dim=0)
                    # Add the last token of the final chunk (which was excluded in the loop condition for the last chunk)
                    tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[-2:-1, :]), dim=0)


            # Make the stacked tensor require gradient for backprop
            tensorStack.requires_grad_(True)
            encoder_outputs = BaseModelOutput(last_hidden_state=tensorStack.to(device).unsqueeze(0),
                                              hidden_states=None, attentions=None)

            # Forward pass with the decoder
            outputs = model(labels=output_ids.to(device).squeeze(0), encoder_outputs=encoder_outputs)
            loss = outputs.loss
            loss = loss / args.grad_accum

            mean_loss += loss.item()
            train_loss += loss.item()

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Retrieve gradients from the encoder's last hidden state
            gradients = outputs.encoder_last_hidden_state.grad
            model.get_encoder().zero_grad() # Zero out encoder gradients for custom accumulation

            # Custom backward pass for the encoder based on how tensorStack was constructed
            indexCount = 0
            if args.dataset_name == "mensa":
                for idx in range(len(scenes)):
                    tokenized_inputs = tokenizer.batch_encode_plus(
                        scenes[idx], truncation=True, padding="do_not_pad",
                        max_length=1024, return_tensors="pt"
                    )
                    # Ensure inputs are moved to device
                    input_ids_chunk = tokenized_inputs["input_ids"].to(device)
                    attention_mask_chunk = tokenized_inputs["attention_mask"].to(device)
                    out = encoder(input_ids=input_ids_chunk, attention_mask=attention_mask_chunk)
                    temp = out[0] # Encoder output for current scene

                    if idx == 0:
                        torch.autograd.backward(temp[:, :-1, :], gradients[:, indexCount:indexCount + temp.shape[1] - 1, :])
                        indexCount = indexCount + temp.shape[1] - 1
                    elif idx < len(scenes) - 1:
                        torch.autograd.backward(temp[:, 1:-1, :], gradients[:, indexCount:indexCount + temp.shape[1] - 2, :])
                        indexCount = indexCount + temp.shape[1] - 2
                    else:
                        torch.autograd.backward(temp[:, 1:, :], gradients[:, indexCount:indexCount + temp.shape[1] - 1, :])
                        indexCount = indexCount + temp.shape[1] - 1
            else:
                # Common logic for other datasets
                chunk_size = 1024 - 2
                if args.dataset_name == "qmsum":
                    chunk_size = 1024 - 2 - query_ids.shape[1]
                last_iteration_index = input_ids.shape[1] - (input_ids.shape[1] % chunk_size)

                for i in range(0, input_ids.shape[1], chunk_size):
                    current_chunk_ids = input_ids[0][i:i + chunk_size].tolist()
                    if args.dataset_name == "qmsum":
                        tokenized_inputs = [bos_token_id] + query_ids[0].tolist() + current_chunk_ids + [eos_token_id]
                    else:
                        tokenized_inputs = [bos_token_id] + current_chunk_ids + [eos_token_id]

                    tokenized_inputs = torch.tensor(tokenized_inputs, dtype=input_ids.dtype).unsqueeze(0).to(device)
                    attention_mask_chunk = torch.ones_like(tokenized_inputs).to(device)
                    out = encoder(input_ids=tokenized_inputs, attention_mask=attention_mask_chunk)
                    temp = out[0]

                    if i == 0:
                        torch.autograd.backward(temp[:, :-1, :], gradients[:, indexCount:indexCount + temp.shape[1] - 1, :])
                        indexCount = indexCount + temp.shape[1] - 1
                    elif i != last_iteration_index:
                        torch.autograd.backward(temp[:, 1:-1, :], gradients[:, indexCount:indexCount + temp.shape[1] - 2, :])
                        indexCount = indexCount + temp.shape[1] - 2
                    else:
                        torch.autograd.backward(temp[:, 1:, :], gradients[:, indexCount:indexCount + temp.shape[1], :])
                        indexCount = indexCount + temp.shape[1] - 1

            if step % args.log_every_step == 0:
                log_metrics(logger, step, {
                    'lr': optimizer.param_groups[0]['lr'],
                    'steps': step,
                    'epochs': epoch,
                    "optimize_steps": completed_steps,
                    'loss/train': loss.item(),
                    "running_train_loss": train_loss,
                    "mean_loss": mean_loss / args.log_every_step
                })
                mean_loss = 0.0

            if step % args.grad_accum == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad() # Zero gradients AFTER optimization step
                completed_steps += 1
                train_loss = 0.0

            # Validation and checkpointing
            if step % args.val_every == 0 and step > 0:
                logger.info(f'Evaluating and saving model at epoch:{epoch} step: {step}')
                val_metrics = evaluate_step(model, args, tokenizer, validation_dataloader,
                                            rouge_metric, bertscore_metric, test=False)
                val_metrics["steps"] = step
                log_metrics(logger, step, val_metrics)
                if val_metrics["val_rouge1"] > best_val_r1:
                    logger.info(f'Metric improved')
                    save_checkpoint(epoch, step, model, args, config, scaler, lr_scheduler, completed_steps, optimizer, best=True)
                    best_val_r1 = val_metrics["val_rouge1"]
                else:
                    save_checkpoint(epoch, step, model, args, config, scaler, lr_scheduler, completed_steps, optimizer, best=False)
                model.train() # Set model back to train mode

            step += 1
            progress_bar.update(1)
            torch.cuda.empty_cache() # Clear CUDA cache

        # Save checkpoint at the end of each epoch
        logger.info(f'Saving model checkpoint at end of epoch:{epoch} step: {step - 1}')
        save_checkpoint(epoch, step, model, args, config, scaler, lr_scheduler, completed_steps, optimizer, best=False)

    logger.info(f'End of training')

def evaluate_step(model, args, tokenizer, data_loader, rouge_metric, bertscore_metric, test=False):
    """
    Evaluates the model on a given dataset.
    Preserves the original evaluation logic including BERTScore and ROUGE calculation.

    Args:
        model (torch.nn.Module): The BART model to evaluate.
        args (argparse.Namespace): Command-line arguments.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        rouge_metric (evaluate.Metric): ROUGE metric object.
        bertscore_metric (evaluate.Metric): BERTScore metric object.
        test (bool, optional): If True, performs test evaluation. Defaults to False (validation).

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    model.eval()

    all_predictions = []
    all_references = []
    metricsDict = {}
    

    for batch in tqdm(data_loader, desc="Evaluating" if test else "Validating"):
        # Determine batch structure based on dataset
        if args.dataset_name == "qmsum":
            input_ids, output_ids, query_ids = batch
        elif args.dataset_name == "mensa":
            scenes, output_ids = batch
            input_ids = None # Not a direct tensor for processing here
        else: # summscreen, govreport, booksum
            input_ids, output_ids = batch

        encoder = model.get_encoder()
        tensorStack = None
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            if args.dataset_name == "mensa":
                for idx in range(len(scenes)):
                    tokenized_inputs = tokenizer.batch_encode_plus(
                        scenes[idx], truncation=True, padding="do_not_pad",
                        max_length=1024, return_tensors="pt"
                    )
                    input_ids_chunk = tokenized_inputs["input_ids"].to(device)
                    attention_mask_chunk = tokenized_inputs["attention_mask"].to(device)

                    out = encoder(input_ids=input_ids_chunk, attention_mask=attention_mask_chunk)
                    if tensorStack is None:
                        tensorStack = out[0].squeeze(0)[:-1, :]
                    else:
                        tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[1:-1, :]), dim=0)
                tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[-2:-1, :]), dim=0)
            else:
                chunk_size = 1024 - 2
                if args.dataset_name == "qmsum":
                    chunk_size = 1024 - 2 - query_ids.shape[1]

                for i in range(0, input_ids.shape[1], chunk_size):
                    current_chunk_ids = input_ids[0][i:i + chunk_size].tolist()
                    if args.dataset_name == "qmsum":
                        tokenized_inputs = [bos_token_id] + query_ids[0].tolist() + current_chunk_ids + [eos_token_id]
                    else:
                        tokenized_inputs = [bos_token_id] + current_chunk_ids + [eos_token_id]

                    tokenized_inputs = torch.tensor(tokenized_inputs, dtype=input_ids.dtype).unsqueeze(0).to(device)
                    attention_mask_chunk = torch.ones_like(tokenized_inputs).to(device)

                    out = encoder(input_ids=tokenized_inputs, attention_mask=attention_mask_chunk)

                    if tensorStack is None:
                        tensorStack = out[0].squeeze(0)[:-1, :]
                    else:
                        tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[1:-1, :]), dim=0)
                tensorStack = torch.cat((tensorStack, out[0].squeeze(0)[-2:-1, :]), dim=0)

            encoder_outputs = BaseModelOutput(last_hidden_state=tensorStack.to(device).unsqueeze(0),
                                              hidden_states=None, attentions=None)

            generated_ids = model.generate(encoder_outputs=encoder_outputs,
                                           use_cache=True, max_length=args.max_output_len, num_beams=args.beam_size)

        # Convert predicted and gold token ids to strings
        predictions = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = tokenizer.batch_decode(output_ids.squeeze(0).tolist(), skip_special_tokens=True)

        predictions, references = postprocess_text(predictions, references)

        all_predictions.extend(predictions)
        all_references.extend(references)

        # Compute rouge and bertscore for current batch
        # Note: evaluate.load() returns metric objects that can accumulate batches.
        # Calling compute() on them computes results for all added batches.
        rouge_metric.add_batch(predictions=predictions, references=references)
        if test: # BERTScore only on test, as per original code
            bertscore_metric.add_batch(predictions=predictions, references=references)

        if not test and args.limit_val_batches != -1 and len(all_predictions) >= args.limit_val_batches:
            break # Stop validation if limit reached

    # Compute final metrics for the whole evaluation set
    metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    results = rouge_metric.compute(use_stemmer=True) # use_stemmer is true for test, so applied to all here.

    for metric_name in metric_names:
        if test:
            metricsDict[f"test_{metric_name}"] = results[metric_name]
        else:
            metricsDict[f"val_{metric_name}"] = results[metric_name]

    if test:
        bert_result = bertscore_metric.compute(lang="en", batch_size=args.batch_size)
        metricsDict["test_bert_p"] = np.mean(bert_result["precision"])
        metricsDict["test_bert_r"] = np.mean(bert_result["recall"])
        metricsDict["test_bert_f"] = np.mean(bert_result["f1"])

        # Save test summaries and predictions
        with open(os.path.join(args.test_summaries, "test.txt"), "a") as f:
            f.write("\n".join(all_predictions))
            f.write("\n====\n")

        with open(os.path.join(args.test_summaries, "pred.pkl"), "wb") as f:
            pickle.dump(all_predictions, f)
        wandb.save(os.path.join(args.test_summaries, "pred.pkl"))

        with open(os.path.join(args.test_summaries, "ref.pkl"), "wb") as f:
            pickle.dump(all_references, f)

    return metricsDict

if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(description="summarization training and evaluation script")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    # Setup directories
    args.checkpoint_path = os.path.join(args.output_dir, args.exp_name, "checkpoints")
    args.test_summaries = os.path.join(args.output_dir, args.exp_name, "test_summaries")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    os.makedirs(args.test_summaries, exist_ok=True)

    set_seed(args.seed)
    logger, run_name = setup_logging(args)

    tokenizer, model, config = load_model_and_tokenizer(args)

    rouge_metric = evaluate.load('rouge')
    bertscore_metric = evaluate.load("bertscore")
    logger.info(f'Using model: {args.model_name}')
    logger.info(f'Training on dataset: {args.dataset_name}')

    # Dynamically select and load the dataset
    dataset_class_map = {
        "qmsum": QMDataset,
        "summscreen": SummScreenDataset,
        "govreport": GovReportDataset,
        "booksum": BookSumDataset,
        "mensa": MensaDataset,
        "summarization_custom": UbadaaSumDataset
    }

    if args.dataset_name not in dataset_class_map:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    DatasetClass = dataset_class_map[args.dataset_name]

    train_dataset = DatasetClass(tokenizer, "train", args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = DatasetClass(tokenizer, "val", args)
    validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_dataset = DatasetClass(tokenizer, "test", args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_training_steps = args.epochs * len(train_dataloader)

    scaler = GradScaler(enabled=args.fp16)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.dataset_name == "qmsum":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup,
                                                        num_training_steps=num_training_steps)
    model.to(device)

    # Start training
    train_model(num_training_steps, model, optimizer, config, tokenizer, scaler, lr_scheduler,
                train_dataloader, validation_dataloader, logger, args, rouge_metric, bertscore_metric)

    # Load best model for final testing
    logger.info(f'Loading the last best model and testing')
    # Make sure 'best_checkpoint.pt' is consistently saved by save_checkpoint when best=True
    model, optimizer, scaler, config, epoch, steps, optimized_steps, lr_scheduler = load_checkpoint(
        model, optimizer, args.fp16, scaler, os.path.join(args.checkpoint_path, "best_checkpoint.pt"), lr_scheduler
    )
    test_metric = evaluate_step(model, args, tokenizer, test_dataloader, rouge_metric, bertscore_metric, test=True)
    print("Testing", test_metric)
    log_test_metrics(logger, test_metric)
    logger.info(f'Training completed')
