from abc import ABC, abstractclassmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.pool import global_mean_pool
import numpy as np

class Decoder(ABC):
    
    @abstractclassmethod
    def decode(self, z, pos_edge_index, **kwargs):
        pass


class SimpleLinearDecoder(nn.Module, Decoder):
    
    def __init__(self,
                 input_dim=8,
                 hidden_dim=128):
        
        super(SimpleLinearDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # learnable parameters dependent on the latent z
        self.fc1 = nn.Linear(hidden_dim, input_dim)
    
    def decode(self, z, edge_index, **kwargs):
        z = z[edge_index[0]] * z[edge_index[1]]
        z = F.relu(self.fc1(z))
        z = torch.matmul(z, z.t())
        # reconstruct the graph using the learned params
        return torch.sigmoid(z) if kwargs['sigmoid'] else z
    
    def forward(self, z, edge_index, **kwargs):
        return self.decode(z, edge_index, **kwargs)
    
    def forward_all(self, z, **kwargs):
        z = F.relu(self.fc1(z))
        z = torch.matmul(z, z.t())
        # reconstruct the graph using the learned params
        return torch.sigmoid(z) if kwargs['sigmoid'] else z
    