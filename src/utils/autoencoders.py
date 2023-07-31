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
    
    def __init__(self, encoder: nn.Module,
                 decoder: Optional[nn.Module] = None,
                 **kwargs):
        
        super().__init__(encoder, decoder)
        
        self.in_dim = kwargs.get('in_dim', 4)
        self.decoder_dims = kwargs.get('decoder_dims', 10)
        self.replace_rate = kwargs.get('replace_rate', .1)
        self.mask_rate = kwargs.get('mask_rate', .3)
        
        print(f'decoder_dims inside the VGAE = {self.decoder_dims}')
        self.encoder_to_decoder = nn.Linear(self.decoder_dims, self.decoder_dims, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.in_dim))
        
        self.mse = nn.MSELoss()
        
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        num_nodes = len(z)
        num_edges = len(pos_edge_index)
        # if the input graph is not complete
        pos_loss = -torch.log(self.decoder.decode(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        neg_loss = 0
        if num_edges != (num_nodes * (num_nodes - 1)):
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            neg_loss = -torch.log(1 - self.decoder.decode(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        return pos_loss + neg_loss
 
    def loss(self, z: Tensor, truth):   
        return self.mse(z, truth) + (1/z.shape[0]) * self.kl_loss()
    
    def encoding_mask_noise(self, x, mask_rate=.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        
        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[:int((1-self.replace_rate) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
            
        out_x[token_nodes] += self.enc_mask_token
        
        return x, mask_nodes, keep_nodes
        
    
    def forward(self, x, edge_index, edge_attr, sigmoid=True):
        use_x, mask_nodes, _ = self.encoding_mask_noise(x, self.mask_rate)
        # encode the x
        z = self.encode(use_x, edge_index, edge_attr)
        # align the encoder latent space with the decoder's input space
        repr = self.encoder_to_decoder(z)
        repr[mask_nodes] = 0
        # get the reconstructed edge probabilities
        recon = self.decoder.decode(repr, edge_index, **{"edge_attr":edge_attr, "sigmoid":sigmoid})
        return recon, z
        
        
class ContrastiveGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None, **kwargs):
        super().__init__(encoder, decoder)
        self.margin = kwargs.get('margin', 2)
        self.contrastive_loss = SupervisedContrastiveLoss(margin=self.margin)
        self.mse = nn.MSELoss()
        
    def loss(self, z, truth):
        z = SimpleNamespace(**z)
        truth = SimpleNamespace(**truth)
        
        g1_repr, g2_repr, z1 = z.g1_repr, z.g2_repr, z.z1_for_rec
        gt, labels = truth.g1_truth, truth.labels
                
        rec_g1 = self.decoder.forward_all(z1, sigmoid=False)
        
        return self.mse(rec_g1, gt), self.contrastive_loss(g1_repr, g2_repr, labels)

        
        
