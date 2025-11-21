import math
import torch
import torch.nn as nn

class GPT2Block(nn.Module):
    """
    Single GPT-2 style Transformer block:

    - LayerNorm -> masked multi-head self-attention -> residual
    - LayerNorm -> MLP (GELU) -> residual
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        attn_mask: (seq_len, seq_len) causal mask
        key_padding_mask: (batch, seq_len) bool mask for padding tokens
        """
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.ln_2(x)
        x = residual + self.mlp(x)

        return x