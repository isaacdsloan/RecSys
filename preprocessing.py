import torch
from torch import nn, Tensor
import torch.optim as optim
from models.VanillaTransformer import VanillaTransformer

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

input_data = 1

output = model(input_data)
reshaped_output = output.reshape(-1, output.shape[2])