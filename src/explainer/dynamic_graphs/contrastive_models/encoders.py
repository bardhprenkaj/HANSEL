from abc import ABC, abstractclassmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Encoder(ABC):
    
    @abstractclassmethod
    def encode(self, graph, edge_index, edge_attr, **kwargs):
        pass


class VariationalGCNEncoder(nn.Module):
    
    def __init__(self,
                 in_channels=1,
                 out_channels=64):
        
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 4 * out_channels)
        self.conv3 = GCNConv(4 * out_channels, 8 * out_channels)
        #self.conv4 = GCNConv(8 * out_channels, 4 * out_channels)
        #self.conv5 = GCNConv(4 * out_channels, 2 * out_channels)
        
        self.conv_mu = GCNConv(8 * out_channels, out_channels)
        self.conv_logstd = GCNConv(8 * out_channels, out_channels)
                
    def forward(self, x, edge_index, edge_weight):
        x = torch.sigmoid(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        """x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))"""
        
        return self.conv_mu(x, edge_index, edge_weight), self.conv_logstd(x, edge_index, edge_weight)
    
    """def encode(self, graph, edge_index, edge_weight, **kwargs):
        with torch.no_grad():
            cosine = torch.nn.CosineSimilarity(dim=0)

            mu, sigma = self.forward(graph, edge_index, edge_weight)
            
            N = torch.distributions.Normal(0, 1)
            z = mu + sigma*N.sample(mu.shape)
            
            encoded_relative_repr = torch.tensor([])
            sampled_points = self.__relative_repr(N, mu, sigma)
            for sample in sampled_points:
                encoded_relative_repr = torch.hstack((encoded_relative_repr, cosine(z.flatten(), sample.flatten())))
            
            return z"""

    def __relative_repr(self, N, mu, sigma, k=4):
        sampled_points = []
        for _ in range(k):
            sampled_points.append(mu + sigma * N.sample(mu.shape))
        return sampled_points
        
    
    def set_training(self, training):
        self.training = training