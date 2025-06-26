import os
import pickle
import torch
from torch.utils.data import Dataset
from datasets import load_dataset # Assuming 'datasets' library is installed and available
import nltk

# Ensure nltk punkt tokenizer is available if not already downloaded
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")


class SummarizationCustomDataset(Dataset):
    """
    Custom Dataset class for summarization tasks loading data from pickle files.
    Used for 'mensa' dataset in original code.
    """
    def __init__(self, tokenizer, split_name, args):
        if split_name == "train":
            self.file_path = os.path.join(args.file_path, "train.pkl")
        elif split_name == "val":
            self.file_path = os.path.join(args.file_path, "val.pkl")
        elif split_name == "test":
            self.file_path = os.path.join(args.file_path, "test.pkl")

        self.tokenizer = tokenizer
        with open(self.file_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an example from the dataset. Input scenes are returned as a list of strings,
        output summary is tokenized.
        """
        entry = self.data[idx]
        scenes = entry["scenes"]
        output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=1024,
                                           padding='max_length', return_tensors="pt")
        return scenes, output_ids

class QMDataset(Dataset):
    """
    Dataset class for QMSum dataset.
    """
    def __init__(self, tokenizer, split_name, args=None):
        qmsum = load_dataset("rohitsaxena/qmsum")

        if split_name == "train":
            self.data = qmsum["train"]
        elif split_name == "val":
            self.data = qmsum["validation"]
        elif split_name == "test":
            self.data = qmsum["test"]

        self.tokenizer = tokenizer

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an example from the dataset. Returns tokenized input, output, and query.
        """
        entry = self.data[idx]
        query = entry["src"].split("</s>")[0].split("<s> ")[1]
        text = entry["src"].split("</s>")[1].strip()

        tokenized_input = self.tokenizer.encode(text, padding="do_not_pad", add_special_tokens=False, return_tensors="pt")
        tokenized_query = self.tokenizer.encode(query, padding="do_not_pad", add_special_tokens=False, return_tensors="pt")

        output_ids = self.tokenizer.encode(entry['tgt'], truncation=True, max_length=1024,
                                           padding='max_length', return_tensors="pt")

        return tokenized_input.squeeze(0), output_ids, tokenized_query.squeeze(0)

class SummScreenDataset(Dataset):
    """
    Dataset class for SummScreen dataset.
    """
    def __init__(self, tokenizer, split_name, args=None):
        summscreen = load_dataset("YuanPJ/summ_screen", "fd")

        if split_name == "train":
            self.data = summscreen["train"]
        elif split_name == "val":
            self.data = summscreen["validation"]
        elif split_name == "test":
            self.data = summscreen["test"]

        self.tokenizer = tokenizer

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an example from the dataset. Returns tokenized transcript and recap.
        """
        entry = self.data[idx]
        tokenized_input = self.tokenizer.encode(" ".join(entry["Transcript"]), padding="do_not_pad", return_tensors="pt")

        output_ids = self.tokenizer.encode(entry['Recap'][0], truncation=True, max_length=1024,
                                           padding='max_length', return_tensors="pt")

        return tokenized_input.squeeze(0), output_ids

class GovReportDataset(Dataset):
    """
    Dataset class for GovReport summarization dataset.
    """
    def __init__(self, tokenizer, split_name, args=None):
        gov_report = load_dataset("ccdv/govreport-summarization")

        if split_name == "train":
            self.data = gov_report["train"]
        elif split_name == "val":
            self.data = gov_report["validation"]
        elif split_name == "test":
            self.data = gov_report["test"]

        self.tokenizer = tokenizer

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an example from the dataset. Returns tokenized report and summary.
        """
        entry = self.data[idx]
        tokenized_input = self.tokenizer.encode(entry["report"], return_tensors="pt")

        output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=1024,
                                           padding='max_length', return_tensors="pt")

        return tokenized_input.squeeze(0), output_ids

class BookSumDataset(Dataset):
    """
    Dataset class for BookSum summarization dataset.
    """
    def __init__(self, tokenizer, split_name, args=None):
        booksum = load_dataset("rohitsaxena/booksum2")

        if split_name == "train":
            self.data = booksum["train"]
        elif split_name == "val":
            self.data = booksum["validation"]
        elif split_name == "test":
            self.data = booksum["test"]

        self.tokenizer = tokenizer

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an example from the dataset. Returns tokenized text and summary.
        """
        entry = self.data[idx]
        tokenized_input = self.tokenizer.encode(entry["text"], padding="do_not_pad", return_tensors="pt")

        output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=1024,
                                           padding='max_length', return_tensors="pt")

        return tokenized_input.squeeze(0), output_ids

class MensaDataset(Dataset):
    """
    Dataset class for rohitsaxena/MENSA dataset.
    It expects 'scenes' (list of strings) and 'summary' columns.
    """
    def __init__(self, tokenizer, split_name, args=None):
        # The rohitsaxena/MENSA dataset structure is assumed to have 'train', 'validation', 'test' splits
        mensa_data = load_dataset("rohitsaxena/MENSA")

        if split_name == "train":
            self.data = mensa_data["train"]
        elif split_name == "val":
            self.data = mensa_data["validation"]
        elif split_name == "test":
            self.data = mensa_data["test"]

        self.tokenizer = tokenizer

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an example from the dataset. Returns list of scenes (strings) and tokenized summary.
        """
        entry = self.data[idx]
        scenes = entry["scenes"]
        output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=1024,
                                           padding='max_length', return_tensors="pt")
        return scenes, output_ids 