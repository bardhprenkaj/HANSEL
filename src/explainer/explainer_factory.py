from typing import List
import torch

from src.dataset.converters.weights_converter import \
    DefaultFeatureAndWeightConverter
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.explainer.dynamic_graphs.explainer_dygrace import DyGRACE
from src.explainer.dynamic_graphs.explainer_gracie import GRACIE
from src.explainer.ensemble.ensemble_factory import EnsembleFactory
from src.explainer.explainer_base import Explainer
from src.explainer.meg.environments.basic_policies import \
    AddRemoveEdgesEnvironment
from src.explainer.meg.environments.bbbp_env import BBBPEnvironment
from src.explainer.meg.explainer_meg import MEGExplainer
from src.explainer.meg.utils.encoders import (
    IDActionEncoder, MorganBitFingerprintActionEncoder)
from src.utils.autoencoder_factory import AEFactory
from src.utils.weight_scheduler_factory import WeightSchedulerFactory
from src.utils.weight_schedulers import WeightScheduler


class ExplainerFactory:

    def __init__(self, explainer_store_path) -> None:
        self._explainer_id_counter = 0
        self._explainer_store_path = explainer_store_path
        self._ensemble_factory = EnsembleFactory(explainer_store_path, self)
        self._autoencoder_factory = AEFactory()
        self._weight_scheduler_factory = WeightSchedulerFactory()

    def get_explainer_by_name(self, explainer_dict, metric_factory : EvaluationMetricFactory) -> Explainer:
        explainer_name = explainer_dict['name']
        explainer_parameters = explainer_dict['parameters']

        # Check if the explainer is DCE Search
        
        if explainer_name == 'dygrace':
            # encoder parameters
            encoder = explainer_parameters['encoder']
            encoder_name = encoder.get('name', 'gcn_encoder')
            if 'parameters' not in encoder:
                raise ValueError('''The encoder of DyGRACE needs to have parameters, even if they're empty''')
            encoder_params = encoder['parameters']
            
            fold_id = explainer_parameters.get('fold_id', 0)
            num_classes = explainer_parameters.get('num_classes', 2)
            in_channels = explainer_parameters.get('input_dim', 1)
            out_channels = explainer_parameters.get('out_dim', 4)
            batch_size = explainer_parameters.get('batch_size', 24)
            lr = explainer_parameters.get('lr', 1e-3)
            epochs_ae = explainer_parameters.get('epochs_ae', 100)
            top_k_cf = explainer_parameters.get('top_k_cf', 10)
            
            encoder = self._autoencoder_factory.get_encoder(encoder_name, **encoder_params)            
            kwargs = {'in_dim': in_channels, 'out_dim': out_channels}
            
            autoencoders = self._autoencoder_factory.init_autoencoders('gae', encoder, None, num_classes, **kwargs)
            
            return self.get_dygrace(fold_id=fold_id,
                                    autoencoders=autoencoders,
                                    num_classes=num_classes,
                                    batch_size=batch_size,
                                    lr=lr,
                                    epochs_ae=epochs_ae,
                                    top_k_cf=top_k_cf,
                                    config_dict=explainer_dict)
            
        elif explainer_name == 'gracie':
            if 'weight_schedulers' not in explainer_parameters:
                raise ValueError('''GRACIE needs to have the weight schedulers specified''')
            if 'decoder' not in explainer_parameters:
                raise ValueError('''GRACIE needs to have a decoder specified''')
            if 'encoder' not in explainer_parameters:
                raise ValueError('''GRACIE needs to have an encoder specified''')
            #########################################################################
            # encoder parameters
            encoder = explainer_parameters['encoder']
            encoder_name = encoder.get('name', 'vgae_')
            if 'parameters' not in encoder:
                raise ValueError('''The encoder of GRACIE needs to have parameters, even if they're empty''')
            encoder_params = encoder['parameters']
            #########################################################################
            # decoder parameters
            decoder = explainer_parameters['decoder']
            decoder_name = decoder.get('name', None)
            if 'parameters' not in decoder:
                raise ValueError('''The decoder of GRACIE needs to have parameters, even if they're empty''')
            decoder_params = decoder['parameters']
            ########################################################################            
            schedulers = explainer_parameters['weight_schedulers']
            # we only need two weight schedulers
            assert(len(schedulers) == 2)
            
            fold_id = explainer_parameters.get('fold_id', 0)
            num_classes = explainer_parameters.get('num_classes', 2)
            batch_size = explainer_parameters.get('batch_size', 24)
            lr = explainer_parameters.get('lr', 1e-3)
            epochs_ae = explainer_parameters.get('epochs_ae', 100)
            top_k_cf = explainer_parameters.get('top_k_cf', 10)
            in_dim = explainer_parameters.get('in_dim', 4)
            replace_rate = explainer_parameters.get('replace_rate', .1)
            mask_rate = explainer_parameters.get('mask_rate', .3)
            lam = explainer_parameters.get('lambda', .5)
                        
            encoder = self._autoencoder_factory.get_encoder(encoder_name, **encoder_params)
            decoder = self._autoencoder_factory.get_decoder(decoder_name, **decoder_params)
            
            kwargs = {'in_dim': in_dim, 'decoder_dims': decoder.in_dim, 'replace_rate': replace_rate, 'mask_rate': mask_rate}
            
            autoencoders = self._autoencoder_factory.init_autoencoders('vgae', encoder, decoder, num_classes, **kwargs)
            
            schedulers = tuple([self._weight_scheduler_factory.get_scheduler_by_name(weight_dict) for weight_dict in schedulers])
            alpha_scheduler, beta_scheduler = schedulers
               
            return self.get_gracie(autoencoders=autoencoders,
                                    alpha_scheduler=alpha_scheduler,
                                    beta_scheduler=beta_scheduler,
                                    batch_size=batch_size,
                                    epochs=epochs_ae,
                                    lr=lr,
                                    k=top_k_cf,
                                    fold_id=fold_id,
                                    lam=lam,
                                    config_dict=explainer_dict)
            
        else:
            raise ValueError('''The provided explainer name does not match any explainer provided 
            by the factory''')
            
            
            
    def get_gracie(self,
                    autoencoders: List[torch.nn.Module],
                    alpha_scheduler: WeightScheduler,
                    beta_scheduler: WeightScheduler,
                    batch_size: int = 8,
                    epochs: int = 100,
                    lr: float = 1e-3,
                    k: int = 10,
                    fold_id: int = 0,
                    lam: int = .5,
                    config_dict = None) -> Explainer:
        
        result = GRACIE(id=self._explainer_id_counter,
                     explainer_store_path=self._explainer_store_path,
                     autoencoders=autoencoders,
                     alpha_scheduler=alpha_scheduler,
                     beta_scheduler=beta_scheduler,
                     batch_size=batch_size,
                     epochs=epochs, lam=lam,
                     lr=lr, k=k, fold_id=fold_id,
                     config_dict=config_dict)
        
        self._explainer_id_counter += 1
        return result  
            
    
    def get_dygrace(self, fold_id:int,
                    autoencoders: List[torch.nn.Module],
                    num_classes: int,
                    batch_size: int,
                    lr: float, epochs_ae: int,
                    top_k_cf: int,
                    config_dict=None):
        
        result = DyGRACE(id=self._explainer_id_counter,
                     explainer_store_path=self._explainer_store_path,
                     fold_id=fold_id,
                     autoencoders=autoencoders,
                     num_classes=num_classes,
                     batch_size=batch_size,
                     lr=lr,
                     epochs_ae=epochs_ae,
                     top_k_cf=top_k_cf,
                     config_dict=config_dict)
        
        self._explainer_id_counter += 1
        return result 