from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

@dataclass
class DataConfig:
    name: str = "tinystories"
    split_train: str = "train"
    split_val: str = "validation"
    max_length: int = 128
    batch_size: int = 64
    num_workers: int = 2

def get_tokenizer() -> AutoTokenizer:
    """
    Use GPT-2 tokenizer (BPE). GPT-2 has no pad token by default,
    so we set pad_token = eos_token for batching convenience.
    """
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok

def load_tinystories_splits(split_train: str, split_val: str):
    """
    TinyStories is commonly available as a HF dataset. Depending on the dataset
    hosting/version, the config name can vary. We'll try a common canonical form.
    """
    df = load_dataset("roneneldan/TinyStories")
    train = df[split_train]
    val = df[split_val] if split_val in df else None
    return train, val

class NextTokenDataset(Dataset):
    """
    Turns raw text examples into fixed-length token sequences.
    For each example we create:
      x = tokens[0:L]
      y = tokens[1:L+1]
    where L = max_length

    We truncate or pad to max_length+1 tokens internally to create x and y.
    """
    def __init__(self, hf_split, tokenizer, max_length: int):
        self.data = hf_split
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        text = item.get("text", None)
        if text is None:
            text = item.get("story","")

        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length + 1,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)

        x = input_ids[:-1].contiguous()
        y = input_ids[1:].contiguous()

        return x,y

def build_dataloaders(cfg: DataConfig):
    tokenizer = get_tokenizer()
    train_split, val_split = load_tinystories_splits(cfg.split_train, cfg.split_val)

    train_ds = NextTokenDataset(train_split, tokenizer, max_length= cfg.max_length)
    val_ds = NextTokenDataset(val_split, tokenizer, max_length=cfg.max_length) if val_split is not None else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, tokenizer