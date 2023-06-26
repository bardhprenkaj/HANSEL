from typing import List

import torch.nn as nn

from src.explainer.dynamic_graphs.contrastive_models.autoencoders import (
    CustomGAE, CustomVGAE)
from src.explainer.dynamic_graphs.contrastive_models.encoders import \
    VariationalGCNEncoder
from src.explainer.dynamic_graphs.contrastive_models.siamese_modules import \
    DenseSiamese


class AEFactory:
           
    def get_model(self,
                  model_name: str,
                  encoder: nn.Module,
                  decoder: nn.Module = None,
                  **kwargs) -> nn.Module:
        
        if model_name.lower() == 'vgae':
            return CustomVGAE(encoder=encoder, decoder=decoder)
        
        elif model_name.lower() == 'gae':
            return CustomGAE(encoder=encoder, decoder=decoder)

        else:
            raise NameError(f"The model name {model_name} isn't supported.")
    
    def get_encoder(self,
                    name,
                    in_channels=1,
                    out_channels=64,
                    **kwargs) -> nn.Module:
        if name.lower() == 'var_gcn_encoder':
            return VariationalGCNEncoder(in_channels=in_channels, out_channels=out_channels)
        else:
            raise NameError(f"The encoder {name} isn't supported.")
        
    def get_decoder(self,
                    name,
                    in_channels=1,
                    out_channels=64,
                    **kwargs) -> nn.Module:
        return None # to be changed in the future. For now InnerProduct is fine.                
    
    
class SiameseFactory:
    
    def get_siamese(self, name, encoders: List[nn.Module], out_channels: int = 4, **kwargs) -> nn.Module:
        if name == 'dense':
            return DenseSiamese(encoders=encoders, out_channels=out_channels, **kwargs)
        else:
            raise NameError(f"The Siamese with name {name} isn't supported.")