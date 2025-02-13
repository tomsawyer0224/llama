import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer"""

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        args:
            dim: dimension of the input tensor (embed_dim)
            eps: a small value added to the denominator for numericall stability
        returns:
            normalized input
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # B, S, E = batch_size, seq_len, embed_dim
        # (B, S, E) * (B, S, 1) = (B, S, E)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        B, S, E = batch_size, seq_len, embed_dim
        args:
            x: (B,S,E)
        returns:
            normalized tensor (B,S,E)
        """
        # weight is a gain parameter used to re-scale the standardized summed inputs
        # (E) * (B, S, E) = (B, S, E)
        return self.weight * self._norm(x.float()).type_as(x)
