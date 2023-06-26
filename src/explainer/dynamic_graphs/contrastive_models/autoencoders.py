from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GAE, VGAE


class AutoEncoder(ABC):
        
    @abstractmethod
    def loss(self, z: Tensor, **kwargs) -> float:
        pass
    
class CustomVGAE(VGAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
        
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        num_nodes = len(z)
        num_edges = len(pos_edge_index[0])
        # if the input graph is not complete
        if num_edges != (num_nodes * (num_nodes - 1)):
            return super().recon_loss(z, pos_edge_index, neg_edge_index)
        else: # i don't need to sample negative edges when the graph is complete
            return -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        
    def loss(self, z: Tensor, **kwargs) -> float:
        loss = self.recon_loss(z, **kwargs)
        loss = loss + (1 / len(z)) * self.kl_loss()
        return loss
    
    
class CustomGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
        
    def loss(self, z: Tensor, **kwargs) -> float:
        return super().recon_loss(z, **kwargs)