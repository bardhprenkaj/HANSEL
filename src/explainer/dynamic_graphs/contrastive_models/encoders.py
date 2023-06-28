from abc import ABC, abstractclassmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class Encoder(ABC):
    
    @abstractclassmethod
    def encode(self, graph, edge_index, edge_attr, **kwargs):
        pass


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, input_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x


class GCNEncoder(nn.Module, Encoder):
    
    def __init__(self,
                 in_channels=1,
                 out_channels=64):
        
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
    
                
    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))   
        return x
    
    def encode(self, x, edge_index, edge_weight):
        with torch.no_grad():
            self.forward(x, edge_index, edge_weight)
    
    def set_training(self, training):
        self.training = training
    
class VariationalGCNEncoder(nn.Module):
    
    def __init__(self,
                 in_channels=1,
                 out_channels=64):
        
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 4 * out_channels)
        
        self.conv_mu = GCNConv(4 * out_channels, out_channels)
        self.conv_logstd = GCNConv(4 * out_channels, out_channels)
                
    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        
        mu = self.conv_mu(x, edge_index, edge_weight)
        sigma = self.conv_logstd(x, edge_index, edge_weight)

        return mu, sigma
    
    def set_training(self, training):
        self.training = training