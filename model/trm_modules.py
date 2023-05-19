import torch.nn as nn
import torch
import math

class Trm_encoder(nn.Module):

    def __init__(self, position_encoder, input_size=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(Trm_encoder, self).__init__()

        # Preprocess
        # self.embedding = nn.Embedding(vocab_size, input_size)
        self.pos_encoder_src = position_encoder(d_model=input_size)
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(input_size, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(input_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()
        self.d_model = input_size
        self.nhead = nhead
        self.sigmoid = nn.Sigmoid()
        # self.linear = nn.Linear(input_size, 1)


    def forward(self, embed_seq, src_mask = None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        if embed_seq.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # position encoding
        embed_seq = self.pos_encoder_src(embed_seq)

        memory = self.encoder(embed_seq, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # memory = self.linear(memory)
        memory = memory.view(memory.size(0), 1, -1)
        memory = nn.Linear(memory.size(-1), 512)
        return memory


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

