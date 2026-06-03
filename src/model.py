"""A small decoder-only Transformer language model, implemented from scratch.

This is a minimal GPT-style character-level language model written with explicit
building blocks (no ``torch.nn.Transformer``) so each part can be explained:

    token embedding + positional embedding
      -> N x [ pre-norm causal multi-head self-attention + residual
               pre-norm feed-forward MLP + residual ]
      -> final layer norm
      -> linear language-modeling head -> logits over the vocabulary

All weights are randomly initialized; nothing here is pretrained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hyperparameters that define the model size and shape.

    Attributes:
        vocab_size: Number of distinct tokens (character ids) the model handles.
        block_size: Maximum context length (number of tokens attended over).
        n_embd: Embedding / hidden dimension.
        n_head: Number of attention heads (must divide ``n_embd``).
        n_layer: Number of stacked Transformer blocks.
        dropout: Dropout probability applied in several places.
    """

    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    Each token may only attend to itself and earlier tokens. Query/key/value
    projections for all heads are computed with a single linear layer and then
    split into ``n_head`` heads.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by "
                f"n_head ({config.n_head})."
            )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Combined projection producing query, key, and value at once.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection after concatenating the heads.
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask, registered as a (non-trained) buffer.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x: Input of shape ``[B, T, n_embd]``.

        Returns:
            Output of shape ``[B, T, n_embd]``.
        """
        B, T, C = x.shape

        # Project to q, k, v and reshape into heads: [B, n_head, T, head_dim].
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: [B, n_head, T, T].
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        # Mask out attention to future positions before softmax.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values, then recombine heads: [B, T, n_embd].
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network (an MLP applied to each token)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP. Input/output shape: ``[B, T, n_embd]``."""
        return self.net(x)


class Block(nn.Module):
    """A single Transformer block using pre-norm residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual attention then residual MLP. Shape preserved: ``[B, T, n_embd]``."""
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """A small decoder-only Transformer language model.

    Maps input token ids ``[B, T]`` to logits ``[B, T, vocab_size]`` and supports
    autoregressive generation.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Randomly initialize weights (from scratch; no pretrained values)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model forward.

        Args:
            idx: Input token ids of shape ``[B, T]`` with ``T <= block_size``.
            targets: Optional target token ids of shape ``[B, T]``. If provided,
                the cross-entropy next-token loss is also returned.

        Returns:
            A tuple ``(logits, loss)`` where ``logits`` has shape
            ``[B, T, vocab_size]`` and ``loss`` is a scalar tensor (or ``None`` if
            no targets were given).

        Raises:
            ValueError: If ``T`` exceeds the configured ``block_size``.
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}."
            )

        pos = torch.arange(T, device=idx.device)
        tok_emb = self.token_embedding(idx)            # [B, T, n_embd]
        pos_emb = self.position_embedding(pos)         # [T, n_embd]
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                       # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate ``max_new_tokens`` tokens.

        Args:
            idx: Conditioning context of shape ``[B, T]``.
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature (>0). Lower is more conservative.
            top_k: If set, restrict sampling to the ``top_k`` most likely tokens.

        Returns:
            A tensor of shape ``[B, T + max_new_tokens]`` containing the context
            followed by the generated tokens.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop the context to the last block_size tokens.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            # Take the logits at the final position and apply temperature.
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
