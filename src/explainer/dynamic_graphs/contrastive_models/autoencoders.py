from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GAE, VGAE
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class AutoEncoder(ABC):
        
    @abstractmethod
    def loss(self, z: Tensor, **kwargs) -> float:
        pass
    
class CustomVGAE(VGAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
            
    def loss(self, z: Tensor, **kwargs) -> float:
        return self.recon_loss(z, **kwargs)
    
class CustomGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
        
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        num_nodes = len(z)
        num_edges = len(pos_edge_index)
        # if the input graph is not complete
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        neg_loss = 0
        if num_edges != (num_nodes * (num_nodes - 1)):
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
            
        return pos_loss + neg_loss
    
    def loss(self, z: Tensor, **kwargs):
        return self.recon_loss(z, **kwargs)