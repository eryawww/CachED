import os
import random
import numpy as np
import torch
import argparse
import logging
import wandb
import tempfile
import shutil
import nltk
from transformers.modeling_outputs import BaseModelOutput # Needed for BaseModelOutput type hint in load_checkpoint

# Ensure nltk punkt tokenizer is available for postprocessing text
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

# Define device globally for convenience, as it's used across multiple functions
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_checkpoint(model, optimizer, fp16, scaler, checkpoint_path, scheduler):
    """
    Loads model, optimizer, scaler, config, epoch, steps, and scheduler from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        fp16 (bool): Whether fp16 (mixed precision) training is enabled.
        scaler (torch.cuda.amp.GradScaler): The GradScaler to load state into, if fp16 is true.
        checkpoint_path (str): Path to the checkpoint file.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to load state into.

    Returns:
        tuple: (model, optimizer, scaler, config, epoch, steps, completed_steps, scheduler)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    config = checkpoint['config']
    epoch = checkpoint['epoch']
    steps = checkpoint['steps']
    completed_steps = checkpoint['completed_steps']
    scheduler.load_state_dict(checkpoint['scheduler'])

    if fp16:
        scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, scaler, config, epoch, steps, completed_steps, scheduler

def set_seed(seed_value=42):
    """
    Set seed for reproducibility across multiple libraries.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def setup_logging(args, project_name="summarization_project"):
    """
    Sets up logging to file and console, and initializes Weights & Biases.

    Args:
        args (argparse.Namespace): Arguments object containing output_dir, exp_name.
        project_name (str): Name of the Weights & Biases project.

    Returns:
        tuple: (logger, run_name) The configured logger and Weights & Biases run name.
    """
    log_dir = os.path.join(args.output_dir, args.exp_name, "log")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"debug_{args.exp_name}.log")),
            logging.StreamHandler()
        ]
    )

    wandb.init(project=project_name, name=args.exp_name, config=args)
    run_name = wandb.run.name

    logger.setLevel(logging.INFO)
    return logger, run_name

def log_test_metrics(logger, metrics):
    """
    Logs final test metrics to console and Weights & Biases.

    Args:
        logger (logging.Logger): The logger instance.
        metrics (dict): Dictionary of test metrics.
    """
    logger.info(f" Final test metrics: {metrics}")
    wandb.log(metrics)

def log_metrics(logger, step, metrics):
    """
    Logs step-wise metrics to console and Weights & Biases.

    Args:
        logger (logging.Logger): The logger instance.
        step (int): Current training step.
        metrics (dict): Dictionary of metrics for the current step.
    """
    logger.info(f"Step {step}: {metrics}")
    wandb.log(metrics)

def save_checkpoint(epoch, step, model, args, config, scaler, scheduler, completed_steps, optimizer, best=False):
    """
    Saves a model checkpoint. Keeps a rotating set of checkpoints and an additional 'best' checkpoint.

    Args:
        epoch (int): Current epoch number.
        step (int): Current global step number.
        model (torch.nn.Module): The model to save.
        args (argparse.Namespace): Arguments object containing checkpoint_path, fp16, max_checkpoints, resume_training.
        config (object): Model configuration.
        scaler (torch.cuda.amp.GradScaler): The GradScaler instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        completed_steps (int): Total completed optimization steps.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        best (bool, optional): If True, also saves this as the best checkpoint. Defaults to False.
    """
    directory = args.checkpoint_path
    os.makedirs(directory, exist_ok=True)

    state = {
        'epoch': epoch,
        'steps': step,
        'config': config,
        'completed_steps': completed_steps,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    if args.fp16:
        state['scaler'] = scaler.state_dict()

    filename = 'checkpoint.pt'
    checkpoint_path = os.path.join(directory, filename)

    with tempfile.NamedTemporaryFile(delete=False) as temp_checkpoint_file:
        torch.save(state, temp_checkpoint_file.name)

    # Implement rotating checkpoints
    root, ext = os.path.splitext(filename)
    for i in range(args.max_checkpoints - 2, -1, -1):
        previous_path = os.path.join(directory, f'{root}{i}{ext}') if i else checkpoint_path
        if os.path.exists(previous_path):
            backup_path = os.path.join(directory, f'{root}{i + 1}{ext}')
            if os.path.exists(backup_path):
                os.replace(previous_path, backup_path)
            else:
                os.rename(previous_path, backup_path)

    shutil.copy(temp_checkpoint_file.name, f'{checkpoint_path}.incomplete')
    os.rename(f'{checkpoint_path}.incomplete', checkpoint_path)
    os.remove(temp_checkpoint_file.name) # Clean up temporary file

    if best:
        best_filename = f'best_checkpoint.pt' # Always use a fixed name for the best checkpoint for easy loading
        best_checkpoint_path = os.path.join(directory, best_filename)
        shutil.copy2(checkpoint_path, best_checkpoint_path)

def add_model_specific_args(parser):
    """
    Adds common model and training arguments to the argparse parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The parser with added arguments.
    """
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-4, help="Maximum learning rate")
    parser.add_argument("--warmup", type=int, default=512, help="Number of warmup steps")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--limit_val_batches", default=-1, type=int, help='Amount of validation data used')
    parser.add_argument("--limit_test_batches", default=0.08, type=float, help='Percent of test data used')
    parser.add_argument("--limit_train_batches", default=0.02, type=float, help='Percent of training data used')
    parser.add_argument("--max_output_len", type=int, default=1024, help="Maximum number of wordpieces in the summary")
    parser.add_argument("--output_dir", type=str, default='/data/output/hirerachical_summ/',
                        help="Location of output directory")
    parser.add_argument("--val_every", default=250, type=int, help='Validation every N steps')

    parser.add_argument("--max_input_len", type=int, default=8192, help="Maximum number of wordpieces in the input")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (fixed to 1 as per request)") # Changed default to 1
    parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 (mixed precision)")
    parser.add_argument("--grad_ckpt", action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment Name")
    parser.add_argument("--checkpoint_path", type=str, default="/rds/user/co-saxe1/hpc-work/output/logging_test/", help="Path to save checkpoints")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="Maximum number of checkpoints to be stored")
    parser.add_argument("--save_checkpoint_steps", type=int, default=20, help="Save checkpoint after N steps")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for generation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--log_every_step", default=10, type=int, help='Logging verbosity (log every N steps)')
    parser.add_argument("--model_name", default="facebook/bart-large-cnn", type=str, help='Hugging Face model to be used')
    parser.add_argument("--resume_training", action='store_true', help='Resume training from best checkpoint')

    # New argument to select dataset
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["mensa", "summscreen", "qmsum", "govreport", "booksum", "summarization_custom"],
                        help="Name of the dataset to train on")
    return parser

def postprocess_text(preds, labels):
    """
    Post-processes predicted and reference texts for ROUGE evaluation.

    Args:
        preds (list): List of predicted summary strings.
        labels (list): List of reference summary strings.

    Returns:
        tuple: (processed_preds, processed_labels)
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
