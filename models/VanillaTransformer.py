import torch as tr
import torch.nn as nn


### Need total number of movies
### Assume inputs are preprocessed


class VanillaTransformer(nn.Module):
    def __init__(
        self, 
        total_num_movies,
        embedding_size,
        input_dim, 
        num_heads, 
        num_encoders,
        max_movie_history_len, 
        ):
        super().__init__()
        self.src_feature_embedding = nn.Embedding(total_num_movies, embedding_size)
        self.src_position_embedding = nn.Embedding(max_movie_history_len, embedding_size)

        self.EncoderLayer = nn.TransformerEncoderLayer(input_dim, num_heads)
        self.EncoderChain = nn.TransformerEncoder(self.EncoderBlock, num_encoders)

        
