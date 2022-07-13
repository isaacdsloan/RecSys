import math
import torch 
from torch import nn, Tensor
from .original_pe import PositionalEncoding

### Need total number of movies
### Assume inputs are preprocessed
### max_movie_history_len <-> vocab_size


class VanillaTransformer(nn.Module):
    def __init__(self, nmovies: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(nmovies, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, nmovies)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, src_pad_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Will zeroed values still remain zero? Should since 0 is not in vocab
        output = self.transformer_encoder(src, src_mask, src_pad_mask)
        output = self.decoder(output)
        return output

    


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



# def __init__(self, total_num_movies, embedding_size, 
#         num_heads, 
#         num_encoders,
#         max_movie_history_len,
#         dim_feedforward,
#         dropout, 
#         device
#         ):
#         super().__init__()
#         self.src_feature_embedding = nn.Embedding(total_num_movies, embedding_size)
#         self.src_position_embedding = nn.Embedding(max_movie_history_len, embedding_size)

#         self.device = device

#         self.EncoderBlock = nn.TransformerEncoderLayer(embedding_size, num_heads, dim_feedforward, dropout)
#         self.EncoderChain = nn.TransformerEncoder(self.EncoderBlock, num_encoders)

#         self.fc_out = nn.Linear(embedding_size, max_movie_history_len)
#         self.dropout = nn.Dropout(dropout)

#         ### pad index?
#         ### add src mask?
# def forward(self, src):
#         input_embed = self.dropout(self.src_feature_embedding(src) + self.src_position_embedding(src))

#         encoder_output = self.EncoderChain(input_embed)

#         ### Returns matrix that is size of input
#         return encoder_output