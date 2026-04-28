"""
Mini-GPT: decoder-only transformer built from scratch with PyTorch.

Components
----------
PositionalEncoding      - sinusoidal fixed encoding (Vaswani et al. 2017)
MultiHeadSelfAttention  - scaled dot-product attention with causal mask
FeedForward             - position-wise two-layer MLP
TransformerBlock        - one decoder layer (attention + FF + residual + LayerNorm)
GPT                     - full language model (embed → N blocks → head)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, DROPOUT


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding added to token embeddings."""

    def __init__(self, embed_dim: int, max_len: int = BLOCK_SIZE) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()         # (T, 1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10_000.0) / embed_dim)
        )                                                          # (embed_dim/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("pe", pe.unsqueeze(0))               # (1, T, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, embed_dim)
        return x + self.pe[:, : x.size(1)]


# ── Multi-Head Self-Attention ─────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Causal (masked) multi-head self-attention.

    Stores the last computed attention weights in self.attn_weights so that
    visualize.py can inspect them without a separate forward pass.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Fused projection for Q, K, V
        self.qkv_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask — upper-triangular (filled at construction, extended if needed)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

        self.attn_weights: torch.Tensor | None = None  # cached for visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V  →  each (B, num_heads, T, head_dim)
        qkv = self.qkv_proj(x)                            # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)      # (B, H, T, head_dim)

        # Scaled dot-product attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_drop(attn)

        self.attn_weights = attn.detach()                  # save for visualization

        # Weighted sum of values
        out = attn @ v                                     # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Position-wise MLP: linear → GELU → linear."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Attention → residual → LayerNorm → FF → residual."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.ff   = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── GPT Language Model ────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Decoder-only transformer language model.

    Parameters
    ----------
    vocab_size : number of unique tokens
    embed_dim  : model width
    num_heads  : attention heads per block
    num_layers : transformer block count
    ff_dim     : feed-forward inner dimension
    block_size : maximum context length
    dropout    : dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int  = EMBED_DIM,
        num_heads:  int  = NUM_HEADS,
        num_layers: int  = NUM_LAYERS,
        ff_dim:     int  = FF_DIM,
        block_size: int  = BLOCK_SIZE,
        dropout:    float = DROPOUT,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=block_size)
        self.drop    = nn.Dropout(dropout)

        self.blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f    = nn.LayerNorm(embed_dim)
        self.head    = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: token embedding and output projection share weights
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx     : (B, T) long tensor of token indices
        targets : (B, T) long tensor, shifted by one position (next-token labels)

        Returns (logits, loss).  loss is None when targets is None.
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        x = self.drop(self.pos_enc(self.tok_emb(idx)))   # (B, T, embed_dim)

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)                            # (B, T, embed_dim)
        logits = self.head(x)                            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def get_attention_weights(self) -> list[torch.Tensor]:
        """Return a list of attention weight tensors, one per block."""
        return [block.attn.attn_weights for block in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        idx:         torch.Tensor,
        max_new:     int,
        temperature: float = 1.0,
        top_k:       int | None = None,
    ) -> torch.Tensor:
        """
        Auto-regressively generate `max_new` tokens after the prompt `idx`.

        idx         : (1, T) seed context
        temperature : >1 = more random, <1 = more deterministic
        top_k       : if set, restrict sampling to the top-k logits
        """
        self.eval()
        for _ in range(max_new):
            ctx    = idx[:, -self.block_size:]               # crop to block_size
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature          # (1, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx    = torch.cat([idx, next_t], dim=1)

        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
