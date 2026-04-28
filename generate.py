"""
Text generation / inference script.

Usage
-----
    python generate.py                              # 200 chars, temp=0.8
    python generate.py --prompt "To be or not"
    python generate.py --max-new 500 --temp 0.5 --top-k 40
    python generate.py --prompt "ROMEO:" --max-new 300 --temp 1.0
"""

import argparse

import torch

from config import CHECKPOINT_PATH, DEVICE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, BLOCK_SIZE, DROPOUT
from data   import download_shakespeare, CharTokenizer
from model  import GPT


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
    print(f"Loaded checkpoint from step {ckpt['step']} (val loss={ckpt['loss']:.4f})")
    return model


def generate(args: argparse.Namespace) -> None:
    text      = download_shakespeare()
    tokenizer = CharTokenizer(text)

    model = load_model(tokenizer.vocab_size)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    idx        = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    print(f"\n{'─'*60}")
    print(f"Prompt : {args.prompt!r}")
    print(f"Temp   : {args.temp}   Top-k : {args.top_k}   New tokens : {args.max_new}")
    print(f"{'─'*60}\n")

    out_ids  = model.generate(idx, max_new=args.max_new, temperature=args.temp, top_k=args.top_k)
    out_text = tokenizer.decode(out_ids[0])

    print(out_text)
    print(f"\n{'─'*60}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with trained mini-GPT")
    p.add_argument("--prompt",  type=str,   default="\n",  help="seed text (default: newline)")
    p.add_argument("--max-new", type=int,   default=200,   help="tokens to generate")
    p.add_argument("--temp",    type=float, default=0.8,   help="sampling temperature")
    p.add_argument("--top-k",   type=int,   default=None,  help="top-k sampling (None = disabled)")
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
