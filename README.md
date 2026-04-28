---
title: Mini-GPT Live
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Watch a 3.17M-param transformer generate Shakespeare live
---

# Mini-GPT Live

A 3.17M-parameter decoder-only transformer (mini-GPT) **built from scratch in pure PyTorch** — no HuggingFace Transformers library, no pretrained weights — and trained on the Tiny Shakespeare corpus.

This Space serves a real-time visualization: watch generated text stream out character-by-character while the attention heatmap and next-token probabilities update live.

## Architecture

| Component | Value |
|---|---|
| Layers | 4 |
| Heads | 8 |
| Embedding dim | 256 |
| Feed-forward dim | 1024 |
| Context window | 128 chars |
| Vocab | 65 chars (character-level) |
| Total params | 3.17M |
| Training | 10,000 steps, val loss = 1.317 |

## What's shown

- **Attention heatmap** — last layer, averaged across heads. Each row is a generated token; each column a position in the context window.
- **Next-token probabilities** — top-10 candidates the model considered before sampling each character.
- **Generated text** — streamed live via Server-Sent Events.

## Tech stack

PyTorch · FastAPI · Server-Sent Events · D3.js · Vanilla JS
