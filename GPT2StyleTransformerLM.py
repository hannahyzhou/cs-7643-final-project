import math
import torch
import torch.nn as nn
from GPT2Block import GPT2Block

class GPTStyle2TransformerLM(nn.Module):
    """
    GPT-2 style decoder-only Transformer language model.

    Architecture:
      - Token embedding + learned positional embedding
      - N x GPT2Block (masked self-attention + MLP)
      - Final LayerNorm
      - LM head (tied weights with token embedding)

    Trained as a causal LM on sequences:
      <bos> instruction <sep> response <eos>
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_idx: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                GPT2Block(
                    d_model=d_model,
                    nhead=nhead,
                    dim_ff=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _make_key_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len)
        Returns boolean mask: True where positions are padding.
        """
        return (x == self.pad_idx)

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """
        Causal mask of shape (seq_len, seq_len) with -inf on future positions.

        This is used as attn_mask for MultiheadAttention so that
        token at position t cannot attend to positions > t.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        )

        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) of token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        key_padding_mask = self._make_key_padding_mask(x)

        batch_size, seq_len = x.size()
        if seq_len > self.max_seq_len:
            x = x[:, : self.max_seq_len]
            seq_len = self.max_seq_len
            key_padding_mask = key_padding_mask[:, : self.max_seq_len]

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = tok_emb + pos_emb

        attn_mask = self._generate_causal_mask(seq_len, x.device)

        for block in self.blocks:
            h = block(
                h,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits