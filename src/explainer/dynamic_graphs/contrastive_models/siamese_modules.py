from typing import List
from src.explainer.dynamic_graphs.contrastive_models.encoders import Encoder

import torch
import torch.nn as nn


class DenseSiamese(nn.Module):
    
    def __init__(self,
                 encoders: List[Encoder],
                 out_channels=4,
                 num_layers_at_bottleneck=4):
        
        super(DenseSiamese, self).__init__()

        self.encoders = encoders
        self.num_layers_at_bottleneck = num_layers_at_bottleneck
        self.in_dimension = out_channels * len(self.encoders)
        
        self.fc_layers = []
        for _ in range(self.num_layers_at_bottleneck - 1):
            self.fc_layers.append(nn.Linear(self.in_dimension, max(self.in_dimension // 2, 2)))
            self.fc_layers.append(nn.ReLU())
            self.in_dimension = max(self.in_dimension // 2, 2)
        self.fc_layers.append(nn.Linear(self.in_dimension, 1))
        
        self.fc_layers = nn.Sequential(*self.fc_layers)
        # freeze encoders
        for i in range(len(self.encoders)):
            self.encoders[i].requires_grad = False
        
        
    def forward(self, graphs, edge_indices, edge_attrs):
        z = torch.tensor([])
        # concatenate all the encodings into a single vector
        for i, encoder in enumerate(self.encoders):
            encoded_graph = encoder.encode(graphs[i], edge_indices[i], edge_attrs[i])
            z = torch.hstack((z, encoded_graph))      
        # pass this vector through all layers
        z = self.fc_layers(z)
        # return the sigmoid of the last layer
        return torch.sigmoid(z.squeeze())
                    