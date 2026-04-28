"""
Attention visualization for the mini-GPT transformer.

Produces two figures:
  1. logs/attention_heads_layerN.png  — heatmap grid (one panel per head) for a chosen layer
  2. logs/attention_all_layers.png    — mean attention collapsed across heads, one panel per layer

Usage
-----
    python visualize.py
    python visualize.py --prompt "To be or not to be" --layer 2
    python visualize.py --prompt "HAMLET:" --all-layers
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from config import (
    CHECKPOINT_PATH, DEVICE,
    EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, BLOCK_SIZE, DROPOUT,
)
from data   import download_shakespeare, CharTokenizer
from model  import GPT


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_model(vocab_size: int) -> GPT:
    model = GPT(
        vocab_size = vocab_size,
        embed_dim  = EMBED_DIM,
        num_heads  = NUM_HEADS,
        num_layers = NUM_LAYERS,
        ff_dim     = FF_DIM,
        block_size = BLOCK_SIZE,
        dropout    = DROPOUT,
    ).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    return model


def run_forward(model: GPT, ids: list[int]) -> list[torch.Tensor]:
    """Run a single forward pass and return per-layer attention weights."""
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        model(idx)
    return model.get_attention_weights()   # list of (1, H, T, T)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _add_token_labels(ax: plt.Axes, tokens: list[str]) -> None:
    T = len(tokens)
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([repr(t)[1:-1] for t in tokens], rotation=90, fontsize=7)
    ax.set_yticklabels([repr(t)[1:-1] for t in tokens], fontsize=7)


def plot_single_layer(
    attn_weights: list[torch.Tensor],
    tokens:       list[str],
    layer:        int,
    out_dir:      str = "logs",
) -> None:
    """Grid of per-head attention heatmaps for a single layer."""
    os.makedirs(out_dir, exist_ok=True)
    W  = attn_weights[layer][0].cpu().numpy()  # (H, T, T)
    H, T, _ = W.shape

    cols = min(H, 4)
    rows = (H + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(rows, cols)

    fig.suptitle(f"Attention Weights — Layer {layer}", fontsize=14, y=1.01)

    for h in range(H):
        ax = axes[h // cols, h % cols]
        im = ax.imshow(W[h, :T, :T], vmin=0, vmax=W[h].max(), cmap="Blues", aspect="auto")
        ax.set_title(f"Head {h}", fontsize=9)
        _add_token_labels(ax, tokens)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(H, rows * cols):
        axes.flat[i].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"attention_heads_layer{layer}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_all_layers(
    attn_weights: list[torch.Tensor],
    tokens:       list[str],
    out_dir:      str = "logs",
) -> None:
    """One heatmap per layer, each averaged over all heads."""
    os.makedirs(out_dir, exist_ok=True)
    T = len(tokens)
    L = len(attn_weights)

    cols = min(L, 4)
    rows = (L + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(rows, cols)

    fig.suptitle("Mean Attention per Layer (all heads averaged)", fontsize=14, y=1.01)

    for l_idx in range(L):
        W  = attn_weights[l_idx][0].cpu().numpy()  # (H, T, T)
        W_mean = W.mean(axis=0)                    # (T, T)
        ax = axes[l_idx // cols, l_idx % cols]
        im = ax.imshow(W_mean[:T, :T], cmap="viridis", aspect="auto")
        ax.set_title(f"Layer {l_idx}", fontsize=9)
        _add_token_labels(ax, tokens)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(L, rows * cols):
        axes.flat[i].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "attention_all_layers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_attention_entropy(
    attn_weights: list[torch.Tensor],
    out_dir:      str = "logs",
) -> None:
    """
    Entropy of each head's attention distribution — a proxy for how focused
    vs. diffuse each head is. Low entropy = sharp / focused; high = spread out.
    """
    os.makedirs(out_dir, exist_ok=True)
    L = len(attn_weights)

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("tab10")

    for l_idx in range(L):
        W = attn_weights[l_idx][0].cpu().numpy()  # (H, T, T)
        H = W.shape[0]
        for h in range(H):
            dist    = W[h]                         # (T, T)
            # Entropy of each row (query position), averaged over positions
            entropy = -(dist * np.log(dist + 1e-9)).sum(axis=-1).mean()
            ax.scatter(l_idx, entropy, color=cmap(h), label=f"H{h}" if l_idx == 0 else None, s=80, zorder=3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean attention entropy (nats)")
    ax.set_title("Attention Head Entropy per Layer")
    ax.set_xticks(range(L))
    ax.legend(title="Head", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "attention_entropy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    text      = download_shakespeare()
    tokenizer = CharTokenizer(text)

    model  = load_model(tokenizer.vocab_size)
    ids    = tokenizer.encode(args.prompt)[: BLOCK_SIZE]
    tokens = [args.prompt[i] for i in range(len(ids))]

    print(f"Prompt  : {args.prompt!r}  ({len(ids)} tokens)")

    attn = run_forward(model, ids)

    if args.all_layers:
        plot_all_layers(attn, tokens)
    else:
        layer = min(args.layer, NUM_LAYERS - 1)
        plot_single_layer(attn, tokens, layer)

    plot_attention_entropy(attn)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize attention weights")
    p.add_argument("--prompt",     type=str,            default="To be, or not to be, that is the question", help="input text")
    p.add_argument("--layer",      type=int,            default=0,     help="which layer to visualize (single-layer mode)")
    p.add_argument("--all-layers", action="store_true",                help="plot mean attention for every layer")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
