from abc import ABC, abstractclassmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATConv

class Decoder(ABC):
    
    @abstractclassmethod
    def decode(self, z, edge_index, **kwargs):
        pass
    
    @abstractclassmethod
    def forward_all(self, z, **kwargs):
        pass


class GATDecoder(nn.Module, Decoder):
    
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 negative_slope):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()

        
        if num_layers == 1:
            self.gat_layers.append(GATConv(
                in_channels=in_dim, out_channels=out_dim, heads=nhead_out,
                concat=False, negative_slope=negative_slope,
                add_self_loops=False
            ))
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_channels=in_dim, out_channels=num_hidden, heads=nhead,
                concat=True, negative_slope=negative_slope,
                add_self_loops=False
            ))
            # hidden layers
            for _ in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    in_channels=num_hidden * self.num_heads,
                    out_channels=num_hidden,
                    heads=nhead,
                    concat=True,
                    negative_slope=negative_slope,
                    add_self_loops=False
                ))
            # output projection
            self.gat_layers.append(GATConv(
                in_channels=num_hidden * self.num_heads,
                out_channels=out_dim,
                heads=nhead_out,
                concat=True,
                negative_slope=negative_slope,
                add_self_loops=False
            ))

        self.head = nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, edge_index, edge_attr):
        hidden_list = []
        for l in range(self.num_layers):
            x, _ = self.gat_layers[l](x, edge_index, edge_attr, return_attention_weights=True)
            hidden_list.append(x)
        x[torch.isnan(x)] = -100
        return self.head(x)
        
    def decode(self, z, edge_index, **kwargs):
        #z = z[edge_index[0]] * z[edge_index[1]]
        recon = self.forward(z, edge_index, kwargs['edge_attr'])
        recon[torch.isnan(recon)] = -100
        return torch.sigmoid(recon) if kwargs['sigmoid'] else recon
    
    def forward_all(self, z, **kwargs):
        recon = self.forward(z, kwargs['edge_index'], kwargs['edge_attr'])        
        recon_edges = torch.matmul(recon, recon.t())
        return torch.sigmoid(recon_edges) if kwargs['sigmoid'] else recon_edges
        
        
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
        z = self.fc1(z)
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