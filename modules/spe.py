import torch
from torch import nn, Tensor
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_query: int, d_key: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term_q = torch.exp(torch.arange(0, d_query, 2) * (-math.log(10000.0) / d_query))
        pe_q = torch.zeros(1, max_len, d_query)
        pe_q[0, :, 0::2] = torch.sin(position * div_term_q)
        pe_q[0, :, 1::2] = torch.cos(position * div_term_q)
        self.register_buffer('pe_q', pe_q, persistent=False)

        div_term_k = torch.exp(torch.arange(0, d_key, 2) * (-math.log(10000.0) / d_key))
        pe_k = torch.zeros(1, max_len, d_key)
        pe_k[0, :, 0::2] = torch.sin(position * div_term_k)
        pe_k[0, :, 1::2] = torch.cos(position * div_term_k)
        self.register_buffer('pe_k', pe_k, persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor]:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        q = q + self.pe_q[:,:q.size(1),:]
        q = self.dropout(q)
        k = k + self.pe_k[:,:k.size(1),:]
        k = self.dropout(k)
        return q, k

class SinusoidalPositionalEncoding_v0(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

