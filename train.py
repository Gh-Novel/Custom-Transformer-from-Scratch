"""
Training script for the mini-GPT transformer.

Usage
-----
    python train.py                     # full training run
    python train.py --iters 1000        # quick smoke-test
    python train.py --resume            # continue from latest checkpoint
"""

import argparse
import os
import time

import torch
import matplotlib.pyplot as plt

from config import (
    EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, BLOCK_SIZE, DROPOUT,
    BATCH_SIZE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, LR, GRAD_CLIP,
    CHECKPOINT_DIR, CHECKPOINT_PATH, LOG_PATH, DEVICE,
)
from data  import get_dataloaders
from model import GPT


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer, step: int, loss: float) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "step":       step,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "loss":       loss,
            "vocab_size": model.tok_emb.num_embeddings,
        },
        CHECKPOINT_PATH,
    )
    print(f"  ✓ Checkpoint saved → {CHECKPOINT_PATH}")


def load_checkpoint(model: GPT, optimizer: torch.optim.Optimizer) -> int:
    ckpt      = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from step {ckpt['step']} (loss={ckpt['loss']:.4f})")
    return ckpt["step"]


@torch.no_grad()
def estimate_loss(
    model:        GPT,
    train_loader,
    val_loader,
    eval_iters:   int = EVAL_ITERS,
) -> dict[str, float]:
    """Average loss over `eval_iters` batches for both splits."""
    model.eval()
    results = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        loader_iter = iter(loader)
        for _ in range(eval_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def plot_loss(train_losses: list, val_losses: list, steps: list) -> None:
    os.makedirs("logs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, train_losses, label="Train", linewidth=2)
    ax.plot(steps, val_losses,   label="Val",   linewidth=2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Mini-GPT Training Loss (Shakespeare)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "logs/loss_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curve saved → {path}")


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    print(f"Device: {DEVICE}")

    train_loader, val_loader, tokenizer = get_dataloaders()

    model = GPT(
        vocab_size = tokenizer.vocab_size,
        embed_dim  = EMBED_DIM,
        num_heads  = NUM_HEADS,
        num_layers = NUM_LAYERS,
        ff_dim     = FF_DIM,
        block_size = BLOCK_SIZE,
        dropout    = DROPOUT,
    ).to(DEVICE)

    print(f"Model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    start_step = 0
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        start_step = load_checkpoint(model, optimizer)

    # Cosine LR scheduler with linear warmup
    warmup_steps = 200
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.iters - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses, logged_steps = [], [], []
    os.makedirs("logs", exist_ok=True)
    log_file = open(LOG_PATH, "a")

    train_iter = iter(train_loader)
    t0 = time.time()

    model.train()
    for step in range(start_step, args.iters):
        # ── Evaluation checkpoint ──────────────────────────────────────────
        if step % args.eval_interval == 0 or step == args.iters - 1:
            losses = estimate_loss(model, train_loader, val_loader)
            elapsed = time.time() - t0
            print(
                f"Step {step:5d} | "
                f"train={losses['train']:.4f}  val={losses['val']:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s elapsed"
            )
            log_file.write(f"{step},{losses['train']:.6f},{losses['val']:.6f}\n")
            log_file.flush()
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            logged_steps.append(step)
            save_checkpoint(model, optimizer, step, losses["val"])

        # ── Forward / backward ─────────────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

    log_file.close()
    plot_loss(train_losses, val_losses, logged_steps)
    print("\nTraining complete.")


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mini-GPT on Shakespeare")
    p.add_argument("--iters",         type=int,   default=MAX_ITERS,      help="training iterations")
    p.add_argument("--eval-interval", type=int,   default=EVAL_INTERVAL,  help="evaluation frequency")
    p.add_argument("--lr",            type=float, default=LR,             help="peak learning rate")
    p.add_argument("--resume",        action="store_true",                 help="resume from checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
