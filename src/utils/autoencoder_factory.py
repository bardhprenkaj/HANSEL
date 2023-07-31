from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn

from src.explainer.dynamic_graphs.contrastive_models.siamese_modules import \
    DenseSiamese
from src.utils.autoencoders import ContrastiveGAE, CustomGAE, CustomVGAE
from src.utils.decoders import SimpleLinearDecoder, GATDecoder
from src.utils.encoders import GCNEncoder, GraphSAGE, VariationalGCNEncoder


class AEFactory:
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
           
    def get_model(self,
                  model_name: str,
                  encoder: nn.Module,
                  decoder: nn.Module = None,
                  **kwargs) -> nn.Module:
        
        if model_name.lower() == 'vgae':
            return CustomVGAE(encoder=encoder, decoder=decoder, **kwargs)        
        if model_name.lower() == 'gae':
            return CustomGAE(encoder=encoder, decoder=decoder)
        if model_name.lower() == 'contrastive_gae':
            return ContrastiveGAE(encoder=encoder, decoder=decoder, **kwargs)
        else:
            raise NameError(f"The model name {model_name} isn't supported.")
    
    def get_encoder(self, name, **kwargs) -> nn.Module:
        kwargs = SimpleNamespace(**kwargs)
        if name.lower() == 'gcn_encoder':
            return GCNEncoder(in_channels=kwargs.input_dim, out_channels=kwargs.out_dim)
        elif name.lower() == 'graph_sage':
            return GraphSAGE(in_channels=kwargs.input_dim, hidden_dim=kwargs.out_dim)
        elif name.lower() == 'var_gcn_encoder':
            return VariationalGCNEncoder(in_channels=kwargs.input_dim, out_channels=kwargs.out_dim)
        else:
            raise NameError(f"The encoder {name} isn't supported.")
        
    def get_decoder(self, name, **kwargs) -> nn.Module:
        kwargs = SimpleNamespace(**kwargs)

        if name == 'simple_linear_decoder':
            return SimpleLinearDecoder(input_dim=kwargs.input_dim,
                                       hidden_dim=kwargs.hidden_dim)
        elif name == 'gat_decoder':
            
            return GATDecoder(in_dim=kwargs.input_dim,
                              num_hidden=kwargs.hidden_dim,
                              out_dim=kwargs.out_dim,
                              num_layers=kwargs.num_layers,
                              nhead_out=kwargs.nhead_out,
                              negative_slope=kwargs.negative_slope,
                              concat_out=kwargs.concat_out)
            
        return None # default to the InnerProduct              
    
    
    def init_autoencoders(self, autoencoder_name: str, encoder: nn.Module, decoder: nn.Module,
                          num_classes: int = 2, **kwargs) -> List[nn.Module]:
        return [
            self.get_model(model_name=autoencoder_name,
                           encoder=encoder,
                           decoder=decoder,
                           **kwargs).double().to(self.device)for _ in range(num_classes)
        ]
    
class SiameseFactory:
    
    def get_siamese(self, name, encoders: List[nn.Module], out_channels: int = 4, **kwargs) -> nn.Module:
        if name == 'dense':
            return DenseSiamese(encoders=encoders, out_channels=out_channels, **kwargs)
        else:
            raise NameError(f"The Siamese with name {name} isn't supported.")