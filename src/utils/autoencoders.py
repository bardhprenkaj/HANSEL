from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Optional

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GAE

from src.utils.losses import SupervisedContrastiveLoss

class AutoEncoder(ABC):
        
    @abstractmethod
    def loss(self, z, truth):
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
        
    def loss(self, z, truth):
        z = SimpleNamespace(**z)
        truth = SimpleNamespace(**truth)
        
        g1_repr, g2_repr, z1 = z.g1_repr, z.g2_repr, z.z1_for_rec
        gt, labels = truth.g1_truth, truth.labels
                
        rec_g1 = self.decoder.forward_all(z1, sigmoid=False)
        #rec_x2 = self.decoder.forward_all(z[1], sigmoid=False)
        
        return self.mse(rec_g1, gt), self.constrastive_loss(g1_repr, g2_repr, labels)
        
        
