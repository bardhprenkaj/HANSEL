from typing import List

import torch.nn as nn
import torch

from src.utils.autoencoders import ContrastiveGAE, CustomGAE
from src.utils.encoders import GCNEncoder, GraphSAGE
from src.explainer.dynamic_graphs.contrastive_models.siamese_modules import \
    DenseSiamese


class AEFactory:
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
           
    def get_model(self,
                  model_name: str,
                  encoder: nn.Module,
                  decoder: nn.Module = None,
                  **kwargs) -> nn.Module:
        
        """if model_name.lower() == 'vgae':
            return CustomVGAE(encoder=encoder, decoder=decoder)"""
        
        if model_name.lower() == 'gae':
            return CustomGAE(encoder=encoder, decoder=decoder)
        if model_name.lower() == 'contrastive_gae':
            return ContrastiveGAE(encoder=encoder, decoder=decoder, **kwargs)
        else:
            raise NameError(f"The model name {model_name} isn't supported.")
    
    def get_encoder(self,
                    name,
                    in_channels=1,
                    out_channels=64,
                    **kwargs) -> nn.Module:
        if name.lower() == 'gcn_encoder':
            return GCNEncoder(in_channels=in_channels, out_channels=out_channels)
        elif name.lower() == 'graph_sage':
            return GraphSAGE(in_channels=in_channels, hidden_dim=out_channels)
        else:
            raise NameError(f"The encoder {name} isn't supported.")
        
    def get_decoder(self,
                    name,
                    in_channels=1,
                    out_channels=64,
                    **kwargs) -> nn.Module:
        return None # to be changed in the future. For now InnerProduct is fine.                
    
    
    def init_autoencoders(self,
                          autoencoder_name: str,
                          enc_name: str,
                          dec_name: str,
                          in_channels: int = 8,
                          out_channels: int = 64,
                          num_classes: int=2,
                          separation_margin=1) -> List[nn.Module]:
        
        return [
            self.get_model(model_name=autoencoder_name,
                           encoder=self.get_encoder(name=enc_name,
                                                    in_channels=in_channels,
                                                    out_channels=out_channels),
                           decoder=self.get_decoder(name=dec_name,
                                                    in_channels=in_channels,
                                                    out_channels=out_channels),
                           margin=separation_margin)\
                               .double().to(self.device)\
                                   for _ in range(num_classes)
        ]
    
class SiameseFactory:
    
    def get_siamese(self, name, encoders: List[nn.Module], out_channels: int = 4, **kwargs) -> nn.Module:
        if name == 'dense':
            return DenseSiamese(encoders=encoders, out_channels=out_channels, **kwargs)
        else:
            raise NameError(f"The Siamese with name {name} isn't supported.")