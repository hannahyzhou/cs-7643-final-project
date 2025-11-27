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

    def get_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor):
        key_padding_mask = (x == self.pad_idx)

        x = x.transpose(0, 1)

        emb = self.pos_encoding(self.token_embedding(x) * math.sqrt(self.d_model))

        src = emb
        tgt = emb

        seq_len = tgt.size(0)
        tgt_mask = self.get_mask(seq_len, tgt.device)

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