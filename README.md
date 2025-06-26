# End-to-End Long Document Summarization using Gradient Caching

This repository contains the implementation for the paper "End-to-End Long Document Summarization using Gradient Caching," accepted in TACL 2025. 

The core contribution of this work lies in an innovative approach to handle extremely long documents by efficiently managing gradients during training, allowing end-to-end summarization without heavily relying on external memory or complex hierarchical models.

## Project Structure

The project is organized into the following key files:

* `src/custom_bart.py`: Contains the modified BART (Bidirectional Encoder Representations from Transformers) model architecture.
* `src/utils.py`: A collection of utility functions essential for the training and evaluation pipeline. This includes checkpoint loading/saving, logging setup (with Weights & Biases integration), argument parsing, seed setting for reproducibility, and text post-processing for metric computation.
* `summarization_datasets.py`: Defines the PyTorch `Dataset` classes for various long document summarization benchmarks. Each class handles data loading and preprocessing specific to its dataset.
    * `MensaDataset` (for `rohitsaxena/MENSA` Hugging Face dataset).
    * `QMDataset` (for QMSum).
    * `SummScreenDataset`.
    * `GovReportDataset`.
    * `BookSumDataset`.
* `train.py`: The main script for training and evaluation process. 

## Setup

Follow these steps to set up the environment and prepare the code for execution.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <your_repo_url>
    # cd <your_repo_name>
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  
    
    pip install -r requirements.txt
    ```
    

3.  **Download NLTK data:**
    The `punkt` tokenizer is required for text post-processing. It's automatically handled in `utils.py` and `datasets.py` but can be manually downloaded if issues arise:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

The `train.py` script is the primary entry point for training and evaluation. You can specify the dataset to train on using the `--dataset_name` argument.

Training
```python

python train.py \
    --dataset_name <DATASET_NAME> \
    --model_name <HUGGINGFACE_MODEL_NAME> \
    --epochs <NUM_EPOCHS> \
    --lr <LEARNING_RATE> \
    --warmup <WARMUP_STEPS> \
    --grad_accum <GRADIENT_ACCUMULATION_STEPS> \
    --fp16 \
    --grad_ckpt \
    --exp_name <EXPERIMENT_NAME> \
    --output_dir <OUTPUT_DIRECTORY> \
    --max_output_len <MAX_SUMMARY_LENGTH> \
    --beam_size <BEAM_SEARCH_SIZE> \
    --val_every <VALIDATION_STEPS> \
    --log_every_step <LOGGING_STEPS> \
    # ... and other arguments as needed (refer to utils.py:add_model_specific_args for full list)
```
### Key Arguments:

`--dataset_name`: Required. Choose from mensa, summscreen, qmsum, govreport, booksum, summarization_custom.

`--epochs`: Number of training epochs.

`--lr`: Learning rate.

`--warmup`: Number of warmup steps for the learning rate scheduler.

`--grad_accum`: Number of gradient accumulation steps (your batch_size is fixed at 1, so this controls effective batch size).

`--fp16`: Use mixed-precision training.

`--grad_ckpt`: Enable gradient checkpointing to save memory.

`--exp_name`: A unique name for your experiment (used for logging and output directories).

`--output_dir`: Base directory for all experiment outputs (logs, checkpoints, test summaries).

`--beam_size`: Beam size for beam search decoding during evaluation.

`--val_every`: Perform validation every N training steps.

`--log_every_step`: Log training metrics every N steps.

`--checkpoint_path`: Directory to save model checkpoints (defaults to inside output_dir/exp_name).

