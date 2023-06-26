import os
from itertools import combinations, product
from operator import itemgetter
from typing import List, Tuple, Dict
from src.dataset.data_instance_base import DataInstance

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.explainer.dynamic_graphs.contrastive_models.factory import AEFactory
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class EVE(Explainer):
    
    def __init__(self,
                 id,
                 explainer_store_path,
                 time=0,
                 num_classes=2,
                 in_channels=1,
                 out_channels=4,
                 fold_id=0,
                 batch_size=24,
                 lr=1e-4,
                 epochs_ae=100,
                 device='cpu',
                 enc_name='var_gcn_encoder',
                 dec_name=None,
                 autoencoder_name='vgae',
                 config_dict=None,
                 **kwargs) -> None:
        
        super().__init__(id, config_dict)
        
        self.time = time
        self.num_classes = num_classes
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.epochs_ae = epochs_ae
        self.device = device
        
        self.iteration = 0
        
        self.ae_factory = AEFactory()
                
        encoder = self.ae_factory.get_encoder(name=enc_name,
                                              in_channels=in_channels,
                                              out_channels=out_channels)
        
        decoder = self.ae_factory.get_decoder(name=dec_name,
                                              in_channels=in_channels,
                                              out_channels=out_channels)
        
        self.autoencoders = [
            self.ae_factory.get_model(model_name=autoencoder_name,
                                      encoder=encoder,
                                      decoder=decoder).double().to(self.device) for _ in range(num_classes)
        ]
        
        self.ae_optimisers = [
            torch.optim.SGD(self.autoencoders[i].parameters(), lr=lr) for i in range(num_classes)
        ]
        
        
        self.contrastive_learner = LogisticRegressionCV(penalty='l2')

        self.explainer_store_path = explainer_store_path
        
        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):        
        self.fit(oracle, dataset, self.fold_id)
        ########################################
        # inference
        data_loaders = self.transform_data(oracle, dataset, index_label=True)
        df = pd.DataFrame()
        for data_loader in data_loaders:
            df = pd.concat([df, pd.DataFrame(
                self.__inference_table(data_loader=data_loader,
                                       dataset=dataset,
                                       oracle=oracle,
                                       search_for_instance=instance))])
            
            
        df.rec_cf *= -1
        
        y_pred = self.contrastive_learner.predict(df.values[:,:-1])
        y_pred_proba = self.contrastive_learner.predict_proba(df.values[:,:-1])
        cf_indices = np.where(y_pred == 1)[0]
        
        print(y_pred)
        print(y_pred_proba)
        
        if len(cf_indices) != 0:
            y_pred_proba = y_pred_proba[cf_indices][:,1].squeeze()
            max_index = np.argmax(y_pred_proba)
            instance = dataset.get_instance(int(df.values[:,-1][cf_indices[max_index]]))
        
        return instance        
        
    def fit(self, oracle: Oracle, dataset: Dataset, fold_id=0):
        explainer_name = f'{self.__class__.__name__}_fit_on_{dataset.name}_fold_id_{fold_id}'
        explainer_uri = os.path.join(self.explainer_store_path, explainer_name)
        self.name = explainer_name
        
        if os.path.exists(explainer_uri):
            # Load the weights of the trained model
            self.load_autoencoders()
            self.load_contrastive_learner(oracle, dataset)
        else:
            # Create the folder to store the oracle if it does not exist
            os.mkdir(explainer_uri)                    
            self.__fit_ae(oracle, dataset)
            self.__fit_linear_objective(oracle, dataset)
            # self.save_explainers()        
        # setting the flag to signal the explainer was already trained
        self._fitted = True    
                      
    def save_autoencoder(self, model, cls):
        torch.save(model.state_dict(),
                 os.path.join(self.explainer_store_path, self.name, f'autoencoder_{cls}'))
        
    def load_autoencoders(self):
        for i, _ in enumerate(self.autoencoders):
            self.autoencoders[i].load_state_dict(torch.load(os.path.join(self.explainer_store_path,
                                                                         self.name, f'autoencoder_{i}')))
            
    def save_constrastive_learner(self):
        constrastive_learner_path = os.path.join(self.explainer_store_path,
                                                 self.name,
                                                 'logistic_regression_weights.joblib')
        
        joblib.dump(self.contrastive_learner, constrastive_learner_path)
        
    def load_contrastive_learner(self, oracle, dataset):
        constrastive_learner_path = os.path.join(self.explainer_store_path,
                                                 self.name,
                                                 'logistic_regression_weights.joblib')
        if os.path.exists(constrastive_learner_path):
            self.contrastive_learner = joblib.load(constrastive_learner_path)
        else:
            self.__fit_linear_objective(oracle, dataset)
    
    def __fit_ae(self, oracle: Oracle, dataset: Dataset):
        data_loaders = self.transform_data(oracle, dataset)

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
                    loss = autoencoder.loss(z, pos_edge_index=edge_index)

                    loss.backward()
                    optimiser.step()
                    
                    losses.append(loss.item())
                
                print(f'Class {cls}, Epoch = {epoch} ----> Loss = {np.mean(losses)}')
                
            self.save_autoencoder(autoencoder, cls)
            
            
    def __fit_linear_objective(self, oracle: Oracle, dataset: Dataset):
        X, y = self.__build_contrastive_table(oracle, dataset)
        self.contrastive_learner.fit(X, y)
        self.save_constrastive_learner()
     
    def __build_contrastive_table(self, oracle: Oracle, dataset: Dataset) -> Tuple[np.array, np.array]:
        pos_data_loader, neg_data_loader = self.transform_data_for_contrastive_learning(oracle, dataset)
        # get positive pairs and transform to dataframe        
        rows: Dict[str, float] = self.__prepare_table(pos_data_loader, dataset, oracle, cls=1.)
        df = pd.DataFrame(rows)
        # get negative pairs and concatenate the two dataframes
        rows = (self.__prepare_table(neg_data_loader, dataset, oracle, cls=0.))
        df = pd.concat([df, pd.DataFrame(rows)])
                    
        df.rec_cf *= -1
        
        X = df.values[:,:-1]
        y = df.values[:,-1]
        
        return X, y
    
    
    def __inference_table(self, data_loader: DataLoader, dataset: Dataset,
                          oracle: Oracle,
                          search_for_instance: DataInstance):
        
        rows = {'rec_cf': [], 'rec_f': [], 'sim': [], 'index': []}
        
            
        for data in data_loader:
            x = data.x.squeeze(dim=0).to(self.device)
            edge_index = data.edge_index.squeeze(dim=0).to(self.device)
            edge_attr = data.edge_attr.squeeze(dim=0).to(self.device)
            
            instance = dataset.get_instance(data.y.item())
            cf_label = oracle.predict(instance)
                        
            for i, autoencoder in enumerate(self.autoencoders):
                z = autoencoder.encode(x, edge_index, edge_attr)
                rec_error  = autoencoder.loss(z, pos_edge_index=edge_index).item()
                
                if cf_label != i:
                    rows['rec_cf'].append(rec_error)
                else:
                    rows['rec_f'].append(rec_error)
                    
            rows['index'].append(data.y.item())
            
            similarity = 1 / (1 + GraphEditDistanceMetric().evaluate(search_for_instance, instance))
            rows['sim'].append(similarity)
            
        return rows
        
    def __prepare_table(self, data_loader: DataLoader, dataset: Dataset, oracle: Oracle, cls=1):
        rows = {'rec_cf': [], 'rec_f': [], 'sim': [], 'y': []}
        for item in data_loader:           
            indices = list(map(lambda label: label.y.item(), item))
            # get the second element
            x = item[-1].x.squeeze(dim=0).to(self.device)
            edge_index = item[-1].edge_index.squeeze(dim=0).to(self.device)
            edge_attr = item[-1].edge_attr.squeeze(dim=0).to(self.device)
            
            instance1 = dataset.get_instance(indices[0])
            instance2 = dataset.get_instance(indices[1])
                        
            for i, autoencoder in enumerate(self.autoencoders):
                z = autoencoder.encode(x, edge_index, edge_attr)
                rec_error = autoencoder.loss(z, pos_edge_index=edge_index).item() 
                
                if i != oracle.predict(instance1):
                    rows['rec_cf'].append(rec_error)
                else:
                    rows['rec_f'].append(rec_error)
                            
            similarity = 1 / (1 + GraphEditDistanceMetric().evaluate(instance1, instance2))
            rows['sim'].append(similarity)
            rows['y'].append(cls)
            
        return rows
            
    def transform_data(self, oracle: Oracle, dataset: Dataset, index_label=False) -> List[DataLoader]:
        features, adj_matrices, _ = self.__get_basic_info(dataset)
        indices = dataset.get_split_indices()[self.fold_id]['train']        

        data_dict_cls = {cls:[] for cls in dataset.get_classes()}
        for i in indices:
            label = oracle.predict(dataset.get_instance(i))                   
            data_dict_cls[label].append(self.__build_geometric_data_instance(adj_matrix=adj_matrices[i],
                                                                             features=features[i],
                                                                             label=label if not index_label else int(i)))
        data_loaders = []
        for cls in data_dict_cls.keys():
            data_loaders.append(DataLoader(
                TorchGeometricDataset(data_dict_cls[cls]),
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
            )
        
        return data_loaders
    
    def transform_data_for_contrastive_learning(self, oracle: Oracle, dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        features, adj_matrices, _ = self.__get_basic_info(dataset)
        # get only the training data to avoid going out of bounds
        indices = dataset.get_split_indices()[self.fold_id]['train'] 
        instances = np.array(dataset.instances, dtype=object)[indices]
        
        split_indices = {label: [] for label in range(self.num_classes)}
        for i, instance in enumerate(instances):
            split_indices[oracle.predict(instance)].append(i)
            
        class_indices = sorted(list(split_indices.values()))
        # transform the entire dataset into torch geometric Data objects
        data = []
        for i in indices:
            data.append(self.__build_geometric_data_instance(adj_matrix=adj_matrices[i],
                                                             features=features[i],
                                                             label=int(i)))
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
            positive_combos[i] = list(itemgetter(*positive_index)(data))
            
        for i, negative_index in enumerate(negative_combos):
            negative_combos[i] = list(itemgetter(*negative_index)(data))
                    
        pos_data_loader = DataLoader(TorchGeometricDataset(positive_combos),
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
        
        neg_data_loader = DataLoader(TorchGeometricDataset(negative_combos),
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=2)
        
        return pos_data_loader, neg_data_loader
    
    
    def __get_basic_info(self, dataset: Dataset, train=False):
        adj_matrices  = [i.to_numpy_array() for i in dataset.instances]
        features = [i.features for i in dataset.instances]
        y = torch.from_numpy(np.array([i.graph_label for i in dataset.instances]))
        
        if train:
            indices = dataset.get_split_indices()[self.fold_id]['train'] 
            
            features = list(itemgetter(*indices)(features))
            adj_matrices = list(itemgetter(*indices)(adj_matrices))
            y =  y[indices]
        
        return features, adj_matrices, y
    
    
    def __build_geometric_data_instance(self, adj_matrix, features, label):
        adj = torch.from_numpy(adj_matrix).double()
        x = torch.from_numpy(features).double()
        w = adj

        a = torch.nonzero(adj)
        w = w[a[:,0], a[:,1]]
        
        return Data(x=x, y=label, edge_index=a.T, edge_attr=w)
            