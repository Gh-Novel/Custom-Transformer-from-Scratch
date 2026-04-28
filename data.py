"""
Dataset utilities for the mini-GPT project.

CharDataset   - character-level tokenizer + PyTorch Dataset
get_dataloaders - builds train/val splits and DataLoaders
download_shakespeare - fetches the Tiny Shakespeare corpus if not cached
"""

import os
import urllib.request
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_URL, DATA_PATH, BLOCK_SIZE, BATCH_SIZE


# ── Download ──────────────────────────────────────────────────────────────────

def download_shakespeare() -> str:
    """Download Tiny Shakespeare to DATA_PATH if absent. Returns the raw text."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Downloading Shakespeare corpus → {DATA_PATH} …")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Done.")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ── Character-Level Tokenizer ─────────────────────────────────────────────────

class CharTokenizer:
    """Bijective mapping between characters and integer token IDs."""

    def __init__(self, text: str) -> None:
        chars         = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi     = {ch: i for i, ch in enumerate(chars)}
        self.itos     = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class CharDataset(Dataset):
    """
    Sliding-window character dataset.

    Each sample is a (input, target) pair where target is input shifted one
    position to the right — the standard language-modelling objective.
    """

    def __init__(self, tokens: list[int], block_size: int = BLOCK_SIZE) -> None:
        self.data       = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk  = self.data[idx : idx + self.block_size + 1]
        x      = chunk[:-1]   # (block_size,)
        y      = chunk[1:]    # (block_size,)  — shifted by one
        return x, y


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_dataloaders(
    val_split: float = 0.1,
    block_size: int  = BLOCK_SIZE,
    batch_size: int  = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    """
    Download corpus, build tokenizer, split into train/val, return DataLoaders.

    Returns
    -------
    train_loader, val_loader, tokenizer
    """
    text      = download_shakespeare()
    tokenizer = CharTokenizer(text)
    tokens    = tokenizer.encode(text)

    split     = int(len(tokens) * (1 - val_split))
    train_ds  = CharDataset(tokens[:split],  block_size)
    val_ds    = CharDataset(tokens[split:],  block_size)

    print(
        f"Vocab size : {tokenizer.vocab_size}\n"
        f"Train tokens: {split:,}  |  Val tokens: {len(tokens) - split:,}\n"
        f"Train batches: {len(train_ds) // batch_size:,}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

    return train_loader, val_loader, tokenizer
