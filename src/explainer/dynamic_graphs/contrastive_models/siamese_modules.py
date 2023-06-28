from typing import List
from src.explainer.dynamic_graphs.contrastive_models.encoders import Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCContrastiveLearner(nn.Module):
    
    def __init__(self, in_dimension=3, num_layers=4):        
        super(FCContrastiveLearner, self).__init__()
        
        self.in_dimension = in_dimension

        self.fc_layers = []
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(self.in_dimension, max(self.in_dimension // 2, 2)))
            self.fc_layers.append(nn.ReLU())
            self.in_dimension = max(self.in_dimension // 2, 2)
            
        self.fc_layers.append(nn.Linear(self.in_dimension, 1))
        
        self.fc_layers = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return torch.sigmoid(x.squeeze())
    
    def predict(self, x):
        with torch.no_grad():
            self.forward(x)
            
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
                    
                    
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive