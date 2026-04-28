"""
FastAPI server that streams mini-GPT generation token-by-token over SSE.

Endpoints
---------
GET  /                  → static frontend (web/index.html)
GET  /api/info          → vocab size, block size, model stats
GET  /api/generate      → SSE stream of tokens with attention + top-k probs

Run:
    uvicorn server:app --reload --port 8000
    # then open http://localhost:8000
"""

import asyncio
import json
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    CHECKPOINT_PATH, DEVICE,
    EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, BLOCK_SIZE, DROPOUT,
)
from data  import download_shakespeare, CharTokenizer
from model import GPT


# ── App & model load ──────────────────────────────────────────────────────────

app = FastAPI(title="Mini-GPT Live")

print("Loading tokenizer & model …")
TEXT      = download_shakespeare()
TOKENIZER = CharTokenizer(TEXT)

MODEL = GPT(
    vocab_size = TOKENIZER.vocab_size,
    embed_dim  = EMBED_DIM,
    num_heads  = NUM_HEADS,
    num_layers = NUM_LAYERS,
    ff_dim     = FF_DIM,
    block_size = BLOCK_SIZE,
    dropout    = DROPOUT,
).to(DEVICE)

if not os.path.exists(CHECKPOINT_PATH):
    raise SystemExit(f"Checkpoint not found at {CHECKPOINT_PATH}. Train first: python train.py")

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
MODEL.load_state_dict(ckpt["model"])
MODEL.eval()

print(f"Model ready  |  step={ckpt['step']}  |  val_loss={ckpt['loss']:.4f}  |  device={DEVICE}")


# ── Streaming generation ──────────────────────────────────────────────────────

@torch.no_grad()
def generate_stream(
    prompt:      str,
    max_new:     int,
    temperature: float,
    top_k:       int | None,
    topk_show:   int = 10,
):
    """
    Yields one dict per generated token containing:
        step       : index of the new token (0-based)
        char       : the sampled character
        topk       : list of [char, prob] for the most-likely next tokens
        attention  : list of floats — what the new token attended to
                     (last layer, averaged across heads, length = current ctx)
        ctx_tokens : list of single-character strings for the current context window
    """
    ids = TOKENIZER.encode(prompt) if prompt else [TOKENIZER.stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Send the prompt itself as step -1 (so the UI can render it before generation)
    yield {
        "type"      : "prompt",
        "text"      : prompt,
        "ctx_tokens": list(prompt),
    }

    for step in range(max_new):
        ctx = idx[:, -BLOCK_SIZE:]
        logits, _ = MODEL(ctx)
        logits_last = logits[0, -1] / max(temperature, 1e-6)

        # Top-k for visualization (BEFORE filtering for sampling)
        full_probs   = F.softmax(logits_last, dim=-1)
        vis_vals, vis_idxs = torch.topk(full_probs, k=min(topk_show, full_probs.size(0)))
        topk_data = [
            [TOKENIZER.itos[i.item()], float(v.item())]
            for v, i in zip(vis_vals, vis_idxs)
        ]

        # Top-k sampling filter
        sample_logits = logits_last.clone()
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(sample_logits, min(top_k, sample_logits.size(0)))
            sample_logits[sample_logits < v[-1]] = float("-inf")

        sample_probs = F.softmax(sample_logits, dim=-1)
        next_t = torch.multinomial(sample_probs, num_samples=1)

        # Attention for the last query position
        attn_layers   = MODEL.get_attention_weights()
        last_layer    = attn_layers[-1][0]              # (H, T, T)
        last_q_attn   = last_layer.mean(dim=0)[-1]      # (T,)

        new_char     = TOKENIZER.itos[next_t.item()]
        ctx_chars    = [TOKENIZER.itos[i] for i in ctx[0].tolist()]

        idx = torch.cat([idx, next_t.unsqueeze(0)], dim=1)

        yield {
            "type"       : "token",
            "step"       : step,
            "char"       : new_char,
            "topk"       : topk_data,
            "attention"  : [float(x) for x in last_q_attn.cpu().tolist()],
            "ctx_tokens" : ctx_chars,
        }

    yield {"type": "done", "total": max_new}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.get("/api/info")
def info():
    return {
        "vocab_size" : TOKENIZER.vocab_size,
        "block_size" : BLOCK_SIZE,
        "embed_dim"  : EMBED_DIM,
        "num_heads"  : NUM_HEADS,
        "num_layers" : NUM_LAYERS,
        "params"     : MODEL.num_parameters(),
        "device"     : str(DEVICE),
        "step"       : ckpt["step"],
        "val_loss"   : float(ckpt["loss"]),
    }


@app.get("/api/generate")
async def generate(
    prompt:  str   = Query("ROMEO:", description="seed text"),
    max_new: int   = Query(200,      ge=1, le=1000),
    temp:    float = Query(0.8,      gt=0, le=2.0),
    top_k:   int   = Query(0,        ge=0, le=200, description="0 = disabled"),
    delay:   float = Query(0.02,     ge=0, le=1.0, description="seconds between tokens"),
):
    """SSE endpoint — streams generation events to the browser."""
    top_k_arg = top_k if top_k > 0 else None

    async def event_stream():
        gen = generate_stream(prompt, max_new, temp, top_k_arg)
        for event in gen:
            yield f"data: {json.dumps(event)}\n\n"
            if delay > 0:
                await asyncio.sleep(delay)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering if proxied
            "Connection": "keep-alive",
        },
    )


# Mount web/ for static asset serving (style.css, app.js)
app.mount("/web", StaticFiles(directory="web"), name="web")
