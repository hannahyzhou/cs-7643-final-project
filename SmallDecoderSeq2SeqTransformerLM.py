import math
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding


class SmallDecoderSeq2SeqTransformerLM(nn.Module):
    """
    Encoder–decoder Transformer LM with a *small* decoder.

    Intended to underperform your GPT-style decoder-only models:
      - Full encoder stack
      - Shallow decoder stack (e.g., 2 layers)
      - Trained with the same next-token objective on
        <bos> instruction <sep> response <eos> sequences.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_idx: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.generator = nn.Linear(d_model, vocab_size)

    def _make_key_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len)
        Returns boolean mask (batch, seq_len) where True marks padding.
        """
        return x == self.pad_idx

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """
        Standard causal mask for decoder self-attention:
        shape (seq_len, seq_len) with -inf on future positions.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)

        We treat the *same* sequence as both source and target:
          - Encoder sees the whole sequence (with padding mask).
          - Small decoder does causal self-attention and cross-attn
            to encoder memory to predict the next token.

        This keeps the interface compatible with your training loop:
            logits = model(input_ids)
        """
        key_padding_mask = self._make_key_padding_mask(x)

        x_t = x.transpose(0, 1)

        emb = self.token_embedding(x_t) * math.sqrt(self.d_model)
        emb = self.pos_encoding(emb)

        src = emb
        tgt = emb

        seq_len = tgt.size(0)
        tgt_mask = self._generate_causal_mask(seq_len, tgt.device)

        memory = self.encoder(
            src,
            src_key_padding_mask=key_padding_mask,
        )

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )

        out = out.transpose(0, 1)
        logits = self.generator(out)
        return logits