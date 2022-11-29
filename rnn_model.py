import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy

class GRUMultiTask(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=64, output_size=4, gru_hidden_size=64, dropout=0.1, 
            gru_num_layer=4, bidirectional=False, decoder_depth=3, decoder_hidden_size = 128):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layer = gru_num_layer
        self.decoder_depth = decoder_depth
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(
            input_size = self.embedding_dim + 1, 
            hidden_size = gru_hidden_size, 
            num_layers = gru_num_layer, 
            bias = True,
            batch_first=True, 
            dropout=dropout,
            bidirectional=bidirectional)
        
        self.num_channel_hidden_out = gru_num_layer
        if bidirectional:
            self.num_channel_hidden_out *= 2
        
        self.size_hidden_out = self.num_channel_hidden_out * gru_hidden_size

        if decoder_depth == 2:
            decoder_mlp = nn.Sequential(
                nn.Linear(self.size_hidden_out, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, output_size)
            )
        elif decoder_depth == 3:
            decoder_mlp = nn.Sequential(
                nn.Linear(self.size_hidden_out, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, output_size)
            )
        elif decoder_depth == 4:
            decoder_mlp = nn.Sequential(
                nn.Linear(self.size_hidden_out, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(decoder_hidden_size, output_size)
            )

        self.task1_mlp = deepcopy(decoder_mlp)
        self.task2_mlp = deepcopy(decoder_mlp)

        
    
    def forward(self, input_data):
        x1 = input_data[0]
        x2 = input_data[1]
        s = input_data[2].cpu()
        # print("before embedding", x1.shape, s.shape)
        # print("x2", x2.shape)
        # print("s", s.shape)
        x1 = self.embeddings(x1)
        # print("after embedding", x1.shape)
        batch_size = x1.shape[0]
        x = torch.cat((x1, x2.unsqueeze(2)), dim=2)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        # print("after pack", x_pack.data.shape)

        _, ht = self.gru(x_pack)
        ht = ht.permute((1,0,2))
        ht = ht.reshape(batch_size, self.size_hidden_out)
        out1 = self.task1_mlp(ht)
        out2 = self.task2_mlp(ht)
        # out_array = torch.hstack((out1, out2)).type(torch.float64)
        # print(out_array.shape)
        # print(out_array.dtype)
        return out1, out2