import math

import pandas as pd
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding


class GPTStyleTransformerLM(nn.Module):
    """
    Encoder-only transformer used as causal language model:
    - Uses TransformerEncoder with a causal (future-masked) src_mask.
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, pad_idx):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.generator = nn.Linear(d_model, vocab_size)

    def get_mask(self, seq_len, device):
        """
        causal mask of shape (seq_len, seq_len) with -inf on future positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask

    def forward(self, x):
        """
        x: (batch, seq_len)
        Returns logits: (batch, seq_len, vocab_size)
        """
        src_key_padding_mask = (x == self.pad_idx)
        x = x.transpose(0, 1)

        emb = self.token_embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoding(emb)

        seq_len = emb.size(0)
        src_mask = self.get_mask(seq_len, emb.device)

        encoded = self.encoder(
            emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoded = encoded.transpose(0, 1)
        logits = self.generator(encoded)
        return logits