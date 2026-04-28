"""
Central configuration for the mini-GPT transformer.
"""
import torch

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"

# ── Model ─────────────────────────────────────────────────────────────────────
BLOCK_SIZE   = 128    # context length (tokens)
EMBED_DIM    = 256    # embedding / model dimension
NUM_HEADS    = 8      # attention heads  (EMBED_DIM must be divisible by NUM_HEADS)
NUM_LAYERS   = 4      # transformer blocks
FF_DIM       = 4 * EMBED_DIM   # feed-forward inner dimension
DROPOUT      = 0.1

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
MAX_ITERS    = 10000
EVAL_INTERVAL= 500
EVAL_ITERS   = 100
LR           = 3e-4
GRAD_CLIP    = 1.0

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH= "checkpoints/gpt_shakespeare.pt"
LOG_PATH       = "logs/loss.txt"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
