import math
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, pad_idx):
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

    def get_tgt_mask(self, tgt_len, device):
        """
        Create mask for target sequence so it can't "peak" at future tokens.
        """
        mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        """
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        # Transformer wants (seq_len, batch, d_model)
        src = src.transpose(0, 1)  # (src_len, batch)
        tgt = tgt.transpose(0, 1)  # (tgt_len, batch)

        # Embed + positional encoding
        src_emb = self.pos_encoding(
            self.token_embedding(src) * math.sqrt(self.d_model)
        )  # (src_len, batch, d_model)
        tgt_emb = self.pos_encoding(
            self.token_embedding(tgt) * math.sqrt(self.d_model)
        )  # (tgt_len, batch, d_model)

        tgt_len = tgt_emb.size(0)
        tgt_mask = self.get_tgt_mask(tgt_len, tgt_emb.device)

        # Encode source
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )  # (src_len, batch, d_model)

        # Decode target, attending to encoder memory
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (tgt_len, batch, d_model)

        out = out.transpose(0, 1)  # (batch, tgt_len, d_model)
        logits = self.generator(out)  # (batch, tgt_len, vocab_size)
        return logits