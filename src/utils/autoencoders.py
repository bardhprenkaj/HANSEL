from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GAE, VGAE
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.pool import global_mean_pool


from src.utils.losses import SupervisedContrastiveLoss


class AutoEncoder(ABC):
        
    @abstractmethod
    def loss(self, z, truth, **kwargs):
        pass
    
class CustomGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__(encoder, decoder)
        
        self.mse = nn.MSELoss()
        
    def loss(self, z: Tensor, truth, **kwargs):
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
        
        self.encoder_to_decoder = nn.Linear(self.decoder_dims, self.decoder_dims, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.in_dim))
        
        self.mse = nn.MSELoss()
        
    def encode(self, *args, **kwargs) -> Tensor:
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=1e+7)
        self.__logstd__[torch.isnan(self.__logstd__)] = .1
        self.__mu__[torch.isnan(self.__mu__)] = 1
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z
    
    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=1e+7)
        logstd[torch.isnan(logstd)] = .1
        mu[torch.isnan(mu)] = 1
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
    def recon_loss(self, z: Tensor,
                   pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None,
                   edge_attr: Optional[Tensor] = None) -> Tensor:
        num_nodes = len(z)
        num_edges = len(pos_edge_index)
        # if the input graph is not complete
        pos_loss = -torch.log(self.decoder.decode(z, pos_edge_index, **{'edge_attr': edge_attr, 'sigmoid': True}) + 1e-15).mean()
        neg_loss = 0
        if num_edges != (num_nodes * (num_nodes - 1)):
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0)).type(torch.int64)
            neg_loss = -torch.log(1 - self.decoder.decode(z, neg_edge_index, **{'edge_attr': None,'sigmoid': True}) + 1e-15).mean()
        return pos_loss + neg_loss
 
    def loss(self, z: Tensor, truth, **kwargs): 
        return self.recon_loss(z, truth, **kwargs) + (1/z.shape[0]) * self.kl_loss()
    
    def encoding_mask_noise(self, x, mask_rate=.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
                
        if self.replace_rate > 0:
            num_noise_nodes = max(int(self.replace_rate * num_mask_nodes), 1)
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
                
        return out_x, mask_nodes, keep_nodes
        
    
    def forward(self, x, edge_index, edge_attr, sigmoid=True):
        use_x, mask_nodes, _ = self.encoding_mask_noise(x, self.mask_rate)
        # encode the x
        z = self.encode(use_x, edge_index, edge_attr)
        # non-existing directed edges make everything be nan
        z[torch.isnan(z)] = -1e-4        
        # align the encoder latent space with the decoder's input space
        repr = self.encoder_to_decoder(z)
        repr[mask_nodes] = 0
        # get the reconstructed node features
        recon = self.decoder.decode(repr, edge_index, **{"edge_attr":edge_attr, "sigmoid":sigmoid})
        return recon, z
        
        
class ContrastiveGAE(GAE, AutoEncoder):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None, **kwargs):
        super().__init__(encoder, decoder)
        self.margin = kwargs.get('margin', 2)
        self.contrastive_loss = SupervisedContrastiveLoss(margin=self.margin)
        self.mse = nn.MSELoss()
        
    def loss(self, z, truth, **kwargs):
        z = SimpleNamespace(**z)
        truth = SimpleNamespace(**truth)
        
        g1_repr, g2_repr, z1 = z.g1_repr, z.g2_repr, z.z1_for_rec
        gt, labels = truth.g1_truth, truth.labels
                
        rec_g1 = self.decoder.forward_all(z1, sigmoid=False)
        
        return self.mse(rec_g1, gt), self.contrastive_loss(g1_repr, g2_repr, labels)

        
class CustomCNNVAE(AutoEncoder, nn.Module):
    
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
                
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)
        
    def loss(self, recon: Tensor, truth, **kwargs):
        kwargs = SimpleNamespace(**kwargs)
        BCE = torch.mean((recon - truth)**2)
        # Kullback-Leibler divergence
        KLD = -0.5 * torch.sum(1 + kwargs.logvar - kwargs.mu.pow(2) - kwargs.logvar.exp())
        return BCE + KLD