from asyncio.windows_events import NULL
from pyparsing import null_debug_action
import torch 
import torch.nn as nn
import torch.optim as optim



### Need total number of movies
### Assume inputs are preprocessed
### max_movie_history_len <-> vocab_size


class VanillaTransformer(nn.Module):
    def __init__(self, total_num_movies, embedding_size, 
        num_heads, 
        num_encoders,
        max_movie_history_len,
        dim_feedforward,
        dropout, 
        device
        ):
        super().__init__()
        self.src_feature_embedding = nn.Embedding(total_num_movies, embedding_size)
        self.src_position_embedding = nn.Embedding(max_movie_history_len, embedding_size)

        self.device = device

        self.EncoderBlock = nn.TransformerEncoderLayer(embedding_size, num_heads, dim_feedforward, dropout)
        self.EncoderChain = nn.TransformerEncoder(self.EncoderBlock, num_encoders)

        self.fc_out = nn.Linear(embedding_size, max_movie_history_len)
        self.dropout = nn.Dropout(dropout)

        ### pad index?
        ### add src mask?
    def forward(self, src):
        input_embed = self.dropout(self.src_feature_embedding(src) + self.src_position_embedding(src))

        encoder_output = self.EncoderChain(input_embed)

        ### Returns matrix that is size of input
        return encoder_output



# Training, may move to seperate file eventually

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

num_epochs = 5
learning_rate = 3e-5
batch_size = 32

###Find
total_num_movies = 100
embedding_size = 512
num_heads = 8
num_encoders = 3
max_movie_history_len = 100
dim_feedforward = 4
dropout = 0.1


model = VanillaTransformer(total_num_movies, embedding_size, 
        num_heads, 
        num_encoders,
        max_movie_history_len,
        dim_feedforward,
        dropout, 
        device).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

input_data = NULL


output = model(input_data)
reshaped_output = output.reshape(-1, output.shape[2])











