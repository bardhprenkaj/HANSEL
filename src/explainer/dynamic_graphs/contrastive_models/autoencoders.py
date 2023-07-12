from abc import ABC, abstractmethod
from typing import Optional
from src.explainer.dynamic_graphs.contrastive_models.losses import SupervisedContrastiveLoss

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GAE, VGAE
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class AutoEncoder(ABC):
        
    @abstractmethod
    def loss(self, z: Tensor, truth):
        pass
    
class CustomGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
        
        self.mse = nn.MSELoss()
        
    def loss(self, z: Tensor, truth):
        rec_adj_matrix = self.decoder.forward_all(z, sigmoid=False)
        return self.mse(rec_adj_matrix, truth)
    
    
class ContrastiveGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None, margin: Optional[float] = 1):
        super().__init__(encoder, decoder)
        
        self.constrastive_loss = SupervisedContrastiveLoss(margin=margin)
        self.mse = nn.MSELoss()
        
    def loss(self, z: Tensor, truth):
        z1, z2 = torch.unbind(z)
        
        rec_x1 = self.decoder.forward_all(z1, sigmoid=False)
        rec_x2 = self.decoder.forward_all(z2, sigmoid=False)
        
        return self.mse(rec_x1, truth[0]), self.constrastive_loss(rec_x1, rec_x2, truth[1])
        
        
