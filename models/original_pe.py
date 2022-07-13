import math
import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        
        self.register_buffer('pe', pe)
        """Buffers are tensors, which are registered in the module and will thus be inside the state_dict.
        These tensors do not require gradients and are thus not registered as parameters.
        This is useful e.g. to track the mean and std in batchnorm layers etc. 
        which should be stored and loaded using the state_dict of the module."""

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            embedding_dim <-> d_model
        """
        x = x + self.pe[:x.size(0)] 
        """Each sequence within a batch is summed by the same 
        positional embedding matrix. The positional embedding matrix
        does not have to be different for every sequence since the
        assumption is every sequence is being looked at 
        independently of the other sequences within the batch"""

        return self.dropout(x)