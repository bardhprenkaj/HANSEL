import os
from itertools import combinations, product
from operator import itemgetter
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.utils.autoencoder_factory import (
    AEFactory, SiameseFactory)
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class AdaptiveGCE(Explainer):
    
    def __init__(self,
                 id,
                 explainer_store_path,
                 time=0,
                 num_classes=2,
                 in_channels=1,
                 out_channels=4,
                 fold_id=0,
                 batch_size=24,
                 lr=1e-3,
                 epochs_ae=100,
                 epochs_siamese=100,
                 device='cpu',
                 enc_name='var_gcn_encoder',
                 dec_name=None,
                 contrastive_base_model='vgae',
                 siamese_name='dense',
                 config_dict=None,
                 **kwargs) -> None:
        
        super().__init__(id, config_dict)
        
        self.time = time
        self.num_classes = num_classes
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.epochs_ae = epochs_ae
        self.epochs_siamese = epochs_siamese
        self.device = device
        
        self.iteration = 0
        
        self.ae_factory = AEFactory()
        self.siamese_factory = SiameseFactory()
        
        encoder = self.ae_factory.get_encoder(name=enc_name,
                                              in_channels=in_channels,
                                              out_channels=out_channels)
        
        decoder = self.ae_factory.get_decoder(name=dec_name,
                                              in_channels=in_channels,
                                              out_channels=out_channels)
        
        self.autoencoders = [
            self.ae_factory.get_model(model_name=contrastive_base_model,
                                      encoder=encoder,
                                      decoder=decoder).double().to(self.device) for _ in range(num_classes)
        ]
        
        self.ae_optimisers = [
            torch.optim.Adam(self.autoencoders[i].parameters(), lr=lr) for i in range(num_classes)
        ]
        
        self.siamese_net = self.siamese_factory.get_siamese(siamese_name,
                                                            [encoder for _ in range(num_classes)],
                                                            out_channels,
                                                            **kwargs).double().to(self.device)
        
        self._fitted = False
        
        self.explainer_store_path = explainer_store_path

        
        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):        
        if(not self._fitted):
            self.fit(oracle, dataset, self.fold_id)
            
        return instance        
        
    def fit(self, oracle: Oracle, dataset: Dataset, fold_id=0):
        explainer_name = f'{self.__class__.__name__}_fit_on_{dataset.name}_fold_id_{fold_id}'
        explainer_uri = os.path.join(self.explainer_store_path, explainer_name)
        self.name = explainer_name
        
        if os.path.exists(explainer_uri):
            # Load the weights of the trained model
            self.load_autoencoders()
            self.load_siamese(oracle, dataset)
        else:
            # Create the folder to store the oracle if it does not exist
            os.mkdir(explainer_uri)                    
            self.__fit(oracle, dataset)
            # self.save_explainers()        
        # setting the flag to signal the explainer was already trained
        self._fitted = True    
        
    def __fit(self, oracle: Oracle, dataset: Dataset):
        self.__fit_ae(dataset)
        self.__fit_siamese(oracle, dataset)                  
                    
    def save_autoencoder(self, model, cls):
        torch.save(model.state_dict(),
                 os.path.join(self.explainer_store_path, self.name, f'autoencoder_{cls}'))
        
    def load_autoencoders(self):
        for i, _ in enumerate(self.autoencoders):
            self.autoencoders[i].load_state_dict(torch.load(os.path.join(self.explainer_store_path,
                                                                         self.name, f'autoencoder_{i}')))
            
    def save_siamese(self, model):
        siamese_save_path = os.path.join(self.explainer_store_path, self.name, f'siamese')
        torch.save(model.state_dict(), siamese_save_path)

    def load_siamese(self, oracle: Oracle, dataset: Dataset):
        siamese_save_path = os.path.join(self.explainer_store_path, self.name, f'siamese')
        if os.path.exists(siamese_save_path):
            self.siamese_net.load_state_dict(torch.load(siamese_save_path))
            self.__init_siamese_weights()
        else:
            self.__fit_siamese(oracle, dataset)
            
    def __fit_ae(self, dataset: Dataset):
        data_loaders = self.transform_data_for_ae(dataset)

        for cls, data_loader in enumerate(data_loaders):
            optimiser = self.ae_optimisers[cls]
            autoencoder = self.autoencoders[cls]
            
            autoencoder.train()
            
            for epoch in range(self.epochs_ae):
                
                losses = []
                for item in data_loader:
                    x = item.x.squeeze(dim=0).to(self.device)
                    edge_index = item.edge_index.squeeze(dim=0).to(self.device)
                    edge_attr = item.edge_attr.squeeze(dim=0).to(self.device)
                    
                    optimiser.zero_grad()
                    
                    z = autoencoder.encode(x, edge_index, edge_attr)
                    loss = autoencoder.loss(z, edge_index=edge_index)

                    loss.backward()
                    optimiser.step()
                    
                    losses.append(loss.item())
                
                print(f'Class {cls}, Epoch = {epoch} ----> Loss = {np.mean(losses)}')
                
            self.save_autoencoder(autoencoder, cls)
    
    def __fit_siamese(self, oracle: Oracle, dataset: Dataset):
        # set the weights of the encoders
        self.__init_siamese_weights()
        pos_data_loader, neg_data_loader = self.transform_data_for_siamese(oracle, dataset)
        
        optimiser = torch.optim.SGD(self.siamese_net.parameters(), lr=1e-3)

        self.siamese_net.train()
                
        for epoch in range(self.epochs_siamese):
            pos_losses = self.__siamese_train_loop(pos_data_loader, optimiser, label=1)
            neg_losses = self.__siamese_train_loop(neg_data_loader, optimiser, label=0)
            
            print(f'Epoch {epoch} ---> Avg loss = {np.mean(pos_losses + neg_losses)}, Pos loss = {np.mean(pos_losses)}, Neg loss = {np.mean(neg_losses)}')
            
    
    def __siamese_train_loop(self, data_loader, optimiser, label=1):
        losses = []
        criterion = nn.BCELoss() 
        for item in data_loader:
            x, edge_index, edge_attr = [], [], []
            for data in item:
                x.append(data[0][1].squeeze(dim=0).to(self.device))
                edge_index.append(data[1][1].squeeze(dim=0).to(self.device))
                edge_attr.append(data[-1][1].squeeze(dim=0).to(self.device))
                    
            optimiser.zero_grad()
                    
            y_pred = self.siamese_net(x, edge_index, edge_attr)
            loss = criterion(y_pred, y_pred.new_full(size=y_pred.shape, fill_value=label))

            loss.backward()
            optimiser.step()
            
            losses.append(loss.item())
            
        return losses
        
        
    def __init_siamese_weights(self):
        for i in range(len(self.autoencoders)):
            self.siamese_net.encoders[i].load_state_dict(self.autoencoders[i].encoder.state_dict())
            
            
    def transform_data_for_siamese(self, oracle: Oracle, dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        features, adj_matrices, _ = self.__get_basic_info(dataset)
        # get only the training data to avoid going out of bounds
        indices = dataset.get_split_indices()[self.fold_id]['train'] 
        instances = np.array(dataset.instances, dtype=object)[indices]
        
        split_indices = {label: [] for label in range(self.num_classes)}
        for i, instance in enumerate(instances):
            split_indices[oracle.predict(instance)].append(i)
            
        class_indices = list(split_indices.values())
        # transform the entire dataset into torch geometric Data objects
        w = None
        a = None
        data = []
        for i in range(len(features)):
            # weights is an adjacency matrix n x n x d
            # where d is the dimensionality of the edge weight vector
            # get all non zero vectors. now the shape will be m x d
            # where m is the number of edges and 
            # d is the dimensionality of the edge weight vector
            adj = torch.from_numpy(adj_matrices[i]).double()
            x = torch.from_numpy(features[i]).double()
            w = adj
            # get the edge indices
            # shape m x 2
            a = torch.nonzero(adj)
            w = w[a[:,0], a[:,1]]
            data.append(Data(x=x, edge_index=a.T, edge_attr=w))
            
        data = np.array(data, dtype=object)
        # generate positive and negative samples
        combos = list(product(*class_indices))
        # filter the combinations to keep only the positive tuples
        positive_combos = set([t for t in combos if len(set(t)) == len(t)])
        # initialise the list of negative tuples
        _max = 0
        for indices in class_indices:
            _max = max(_max, max(indices))
            
        negative_combos = set(combinations(range(_max + 1), len(class_indices)))
        negative_combos = negative_combos.difference(positive_combos)
        
        positive_combos = list(map(list, positive_combos))
        negative_combos = list(map(list, negative_combos))
        
        for i, positive_index in enumerate(positive_combos):
            positive_combos[i] = data[positive_index].tolist()
            
        for i, negative_index in enumerate(negative_combos):
            negative_combos[i] = data[negative_index].tolist()
            
        pos_data_loader = DataLoader(TorchGeometricDataset(positive_combos),
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
        
        neg_data_loader = DataLoader(TorchGeometricDataset(negative_combos),
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=2)
        
        return pos_data_loader, neg_data_loader
                
        
    def transform_data_for_ae(self, dataset: Dataset) -> List[DataLoader]:
        features, adj_matrices, y = self.__get_basic_info(dataset)
        
        classes = dataset.get_classes()

        data_dict_cls = {cls:[] for cls in classes}
        w = None
        a = None
        for i in range(len(y)):
            # weights is an adjacency matrix n x n x d
            # where d is the dimensionality of the edge weight vector
            # get all non zero vectors. now the shape will be m x d
            # where m is the number of edges and 
            # d is the dimensionality of the edge weight vector
            adj = torch.from_numpy(adj_matrices[i]).double()
            x = torch.from_numpy(features[i]).double()
            w = adj
            # get the edge indices
            # shape m x 2
            a = torch.nonzero(adj)
            w = w[a[:,0], a[:,1]]
                                            
            data_dict_cls[y[i].item()].append(Data(x=x, y=y[i], edge_index=a.T, edge_attr=w))            
        
        data_loaders = []
        for cls in data_dict_cls.keys():
            data_loaders.append(DataLoader(
                TorchGeometricDataset(data_dict_cls[cls]),
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
            )
        
        return data_loaders
    
    
    def __get_basic_info(self, dataset: Dataset):
        adj_matrices  = [i.to_numpy_array() for i in dataset.instances]
        features = [i.features for i in dataset.instances]
        y = torch.from_numpy(np.array([i.graph_label for i in dataset.instances]))
        
        indices = dataset.get_split_indices()[self.fold_id]['train'] 
        
        features = list(itemgetter(*indices)(features))
        adj_matrices = list(itemgetter(*indices)(adj_matrices))
        y =  y[indices]
        
        return features, adj_matrices, y