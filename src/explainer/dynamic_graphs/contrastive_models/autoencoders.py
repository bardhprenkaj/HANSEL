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
    def loss(self, z: Tensor, truth: Tensor) -> float:
        pass
    
"""class CustomVGAE(VGAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
            
    def loss(self, z: Tensor, **kwargs) -> float:
        loss = self.recon_loss(z, **kwargs)
        loss = loss + (1 / z.shape[0]) * self.kl_loss()
        return loss"""
    
class CustomGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
        
        self.mse = nn.MSELoss()
        
    def loss(self, z: Tensor, truth: Tensor):
        rec_adj_matrix = self.decoder.forward_all(z, sigmoid=False)
        return self.mse(rec_adj_matrix, truth)
        
        
