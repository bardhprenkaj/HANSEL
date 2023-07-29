from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GAE, VGAE
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import negative_sampling


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
    
class CustomVGAE(VGAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None, margin=.5):
        super().__init__(encoder, decoder)
        self.kl_weight = margin
        
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
 
    def loss(self, z: Tensor, truth):   
        return self.recon_loss(z, pos_edge_index=truth) + (1/z.shape[0]) * self.kl_loss()
        
class ContrastiveGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None, margin: Optional[float] = 1):
        super().__init__(encoder, decoder)
        
        self.contrastive_loss = SupervisedContrastiveLoss(margin=margin)
        self.mse = nn.MSELoss()
        
    def loss(self, z, truth):
        z = SimpleNamespace(**z)
        truth = SimpleNamespace(**truth)
        
        g1_repr, g2_repr, z1 = z.g1_repr, z.g2_repr, z.z1_for_rec
        gt, labels = truth.g1_truth, truth.labels
                
        rec_g1 = self.decoder.forward_all(z1, sigmoid=False)
        
        return self.mse(rec_g1, gt), self.contrastive_loss(g1_repr, g2_repr, labels)

        
        
