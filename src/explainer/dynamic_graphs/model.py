import os
from copy import deepcopy
from itertools import combinations, permutations, product
from operator import itemgetter
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.explainer.dynamic_graphs.contrastive_models.factory import AEFactory
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class DyGRACE(Explainer):
    
    def __init__(self,
                 id,
                 explainer_store_path,
                 num_classes=2,
                 in_channels=1,
                 out_channels=4,
                 fold_id=0,
                 batch_size=24,
                 lr=1e-4,
                 epochs_ae=100,
                 device='cpu',
                 enc_name='gcn_encoder',
                 dec_name=None,
                 autoencoder_name='gae',
                 config_dict=None,
                 **kwargs) -> None:
        
        super().__init__(id, config_dict)
        
        self.num_classes = num_classes
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.epochs_ae = epochs_ae
        self.device = device
        self.lr = lr
        
        self.iteration = 0
        
        self.ae_factory = AEFactory()
        
        self.autoencoders = [
            self.ae_factory.get_model(model_name=autoencoder_name,
                                      encoder=self.ae_factory.get_encoder(name=enc_name,
                                                                          in_channels=in_channels,
                                                                          out_channels=out_channels),
                                      decoder=self.ae_factory.get_decoder(name=dec_name,
                                                                          in_channels=in_channels,
                                                                          out_channels=out_channels))\
                                                                              .double().to(self.device)\
                                                                                  for _ in range(num_classes)
        ]
        
        self.contrastive_learner = LogisticRegression()        
        
        self.explainer_store_path = explainer_store_path
        
        self.EPS = 20
        
        self.K = 5

        
        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        explainer_name = f'{self.__class__.__name__}_fit_on_{dataset.name}_fold_id_{self.fold_id}'
        self.explainer_uri = os.path.join(self.explainer_store_path, explainer_name)
        self.name = explainer_name
        
        # train using the oracle only in the first iteration
        if self.iteration == 0:        
            self.fit(oracle, dataset, self.fold_id)
        ########################################
        # inference
        data_loaders = self.transform_data(oracle, dataset, index_label=True, use_oracle=False)
        df = pd.DataFrame(columns=['rec_cf', 'rec_f', 'sim', 'instance'])

        for data_loader in data_loaders:
            rows = self.__inference_table(data_loader=data_loader,
                                          dataset=dataset,
                                          search_for_instance=instance)
            df = pd.concat([df, pd.DataFrame(rows)])
                
        df.rec_cf *= -1
                        
        y_pred = self.contrastive_learner.predict(df.values[:,:-1])
        y_pred = np.where(y_pred == 1)[0]
                
        counterfactuals = []
        if len(y_pred) != 0:
            y_pred_df = pd.DataFrame(self.contrastive_learner.predict_proba(df.values[:,:-1])[:,1], columns=['probability'])
            y_pred_df.sort_values(by='probability', ascending=False, inplace=True)
            
            best_cf = y_pred_df.index[:self.K]
                       
            counterfactuals = set([dataset.get_instance(int(i)) for i in df.values[:,-1][best_cf]])
            counterfactuals = list(counterfactuals)
        else:
            counterfactuals = [instance]
        
        if self.iteration > 0:
            self.update(deepcopy(instance), deepcopy(counterfactuals), oracle)            

        return counterfactuals
        
    def fit(self, oracle: Oracle, dataset: Dataset, fold_id=0):
        if os.path.exists(self.explainer_uri):
            # Load the weights of the trained model
            self.load_autoencoders()
            self.load_contrastive_learner(oracle, dataset)
        else:
            self.__fit_ae(oracle, dataset)
            indices = dataset.get_split_indices()[self.fold_id]['train'] 
            self.__fit_linear_objective(oracle, dataset, indices, use_oracle=True)
            # self.save_explainers()        
        # setting the flag to signal the explainer was already trained
        self._fitted = True    
                      
    def save_autoencoder(self, model, cls):
        try:
            os.mkdir(self.explainer_uri)                    
        except FileExistsError:
            print(f'{self.explainer_uri} already exists. Proceeding with saving weights.')
            
        torch.save(model.state_dict(),
                 os.path.join(self.explainer_store_path, self.name, f'autoencoder_{cls}'))
        
    def load_autoencoders(self):
        for i, _ in enumerate(self.autoencoders):
            self.autoencoders[i].load_state_dict(torch.load(os.path.join(self.explainer_store_path,
                                                                         self.name, f'autoencoder_{i}')))
            
    def save_constrastive_learner(self):
        constrastive_learner_path = os.path.join(self.explainer_store_path,
                                                 self.name,
                                                 f'{self.contrastive_learner.__class__.__name__}_weights.joblib')
        
        joblib.dump(self.contrastive_learner, constrastive_learner_path)
        
    def load_contrastive_learner(self, oracle, dataset):
        constrastive_learner_path = os.path.join(self.explainer_store_path,
                                                 self.name,
                                                 f'{self.contrastive_learner.__class__.__name__}_weights.joblib')
        if os.path.exists(constrastive_learner_path):
            self.contrastive_learner = joblib.load(constrastive_learner_path)
        else:
            indices = dataset.get_split_indices()[self.fold_id]['train'] 
            self.__fit_linear_objective(oracle, dataset, indices, use_oracle=True)
               
    def __fit_ae(self, oracle: Oracle, dataset: Dataset):
        data_loaders = self.transform_data(oracle, dataset)
        for cls, data_loader in enumerate(data_loaders):
            self.__train(data_loader, cls)
            
    def update(self, instance: DataInstance, counterfactuals: List[DataInstance], oracle):
        # the contrastive learner fails to produce a valid counterfactual
        # in these cases, just save the previous autoencoders and contrastive learner
        if len(counterfactuals) == 1 and counterfactuals[0] == instance:
            for cls in range(len(self.autoencoders)):
                self.save_autoencoder(self.autoencoders[cls], cls)
            self.save_constrastive_learner()
            return
        
        x, edge_index, edge_weights, _, truth = self.__expand(self.__to_geometric(instance, label=torch.Tensor([-1])))
        # get the reconstruction errors of each autoencoder for the input instace
        with torch.no_grad():
            rec_errors = [autoencoder.loss(autoencoder.encode(x, edge_index, edge_weights), truth) for autoencoder in self.autoencoders]
        # the lowest reconstruction erorr will represent the factual autoencoder    
        factual_label = np.argmin(rec_errors)
        # transform the list of counterafactuals in a torch geometric data loader
        counterfactuals_geometric = [self.__to_geometric(counterfactual, label=torch.Tensor([-1])) for counterfactual in counterfactuals]
        data_loader = GeometricDataLoader(TorchGeometricDataset(counterfactuals_geometric), batch_size=1, shuffle=True, num_workers=2)
        # set the labels to the instance and counterfactuals
        # in a semi-supervised fashion via the learned autoencoders
        instance.graph_label = factual_label
        # the label of the counterfactuals is any one different from factual_label
        for i in range(len(counterfactuals)):
            counterfactuals[i].graph_label = 1 - factual_label
        # with the counterfactuals found now:
        # 1. train the counterfactual autoencoder and minimise the reconstruction error
        # 2. train the factual autoencoder and maximise the reconstruction error
        for i in range(len(self.autoencoders)):
            if i != factual_label:
                self.__train(data_loader, i)
            #else:
            #    self.__train(data_loader, i, contrastive=False)
            self.save_autoencoder(self.autoencoders[i], i)
        # update the linear objective
        oracle = None # we don't need the oracle in the next iterations
        # build a dummy dataset with the instance and its counterfactuals
        inference_dataset = Dataset(id=-1)
        inference_dataset.instances = [instance] + counterfactuals
        
        # update the contrastive learner
        self.__fit_linear_objective(oracle, inference_dataset, list(range(len(inference_dataset.instances))), use_oracle=False, partial_fit=True)
            
    def __fit_linear_objective(self, oracle: Oracle, dataset: Dataset, indices, use_oracle=False, partial_fit=False):
        X, y = self.__build_contrastive_table(oracle, dataset, indices, use_oracle=use_oracle)
        if len(X):
            if partial_fit:
                self.contrastive_learner.fit(X,y)
            else:
                self.contrastive_learner.partial_fit(X,y)
            print(f'Training R2 = {self.contrastive_learner.score(X, y)}')
            self.save_constrastive_learner()
     
    def __build_contrastive_table(self, oracle: Oracle, dataset: Dataset, indices: np.array, use_oracle=True) -> Tuple[np.array, np.array]:
        pos_data_loader, neg_data_loader = self.__contrastive_learning(oracle, dataset, indices, use_oracle=use_oracle)
        if pos_data_loader == None or neg_data_loader == None:
            return [], []
        # get positive pairs and transform to dataframe        
        rows: List[Dict[str, float]] = self.__training_table(pos_data_loader, dataset, oracle, cls=1., use_oracle=use_oracle)
        df = pd.DataFrame(rows)
        # get negative pairs and concatenate the two dataframes
        rows = (self.__training_table(neg_data_loader, dataset, oracle, cls=0., use_oracle=use_oracle))
        df = pd.concat([df, pd.DataFrame(rows)])
        # get the y label
        y = df.y.values
        # drop it
        df.drop(columns=['y'], inplace=True)
        # ignore the "tuple" column
        X = df.values[:,1:]
        return X, y
    
    
    def __inference_table(self, data_loader: GeometricDataLoader, dataset: Dataset,
                          search_for_instance: DataInstance):
        
        rows = {'rec_cf': [], 'rec_f': [], 'sim': [], 'instance': []}
        
        with torch.no_grad():
            
            for data in data_loader:
                x, edge_index, edge_attr, label, truth = self.__expand(data)

                instance = dataset.get_instance(label.item())                
                # get the combinations of autoencoders to
                # calculate factual and counterfactual reconstruction errors
                combos_autoencoders = combinations(self.autoencoders, r=len(self.autoencoders))
                combos_autoencoders = set(self.__get_tuple_permutations(combos_autoencoders))

                for f_ae, cf_ae in combos_autoencoders:
                    z = f_ae.encode(x, edge_index=edge_index, edge_weight=edge_attr)
                    rec_error = f_ae.loss(z, truth).item()
                    rows['rec_f'].append(rec_error)
                    
                    z = cf_ae.encode(x, edge_index=edge_index, edge_weight=edge_attr)
                    rec_error = cf_ae.loss(z, truth).item()
                    rows['rec_cf'].append(rec_error)
                        
                rows['instance'] += [instance.id] * len(self.autoencoders)
                
                similarity = 1 / (1 + GraphEditDistanceMetric().evaluate(search_for_instance, instance))
                rows['sim'] += [similarity] * len(self.autoencoders)
            
        return rows
        
    def __training_table(self, data_loader: GeometricDataLoader, dataset: Dataset, oracle: Oracle, cls=1, use_oracle=True):
        rows = []
        
        with torch.no_grad():
            for item in data_loader:           
                indices = list(map(lambda label: label.y.item(), item))
                # get the anchor instance and its predicted label
                anchor_instance = dataset.get_instance(indices[0])
                anchor_label = oracle.predict(anchor_instance) if use_oracle else anchor_instance.graph_label
                                
                curr_row = {'tuple': [], 'y': -1, 'rec_cf': 0, 'rec_f': 1}
                # set dummy reconstruction errors for all elements in the tuple
                # set dummy similarity for all elements in the tuple
                curr_row.update({f'sim_{i}': 0 for i in range(len(indices) - 1)})
                # for all the other instances in the tuple
                # get the reconstruction errors (counterfactual and factual)
                # and their similarity with the anchor
                for i, data in enumerate(item[1:]):
                    x, edge_index, edge_attr, _, truth = self.__expand(data)
                    # get the current instance
                    instance = dataset.get_instance(indices[i + 1])
                    # measure reconstruction error                                        
                    for j, autoencoder in enumerate(self.autoencoders):
                        # encode the current graph and take its reconstruction error
                        z = autoencoder.encode(x, edge_index=edge_index, edge_weight=edge_attr)
                        rec_error = autoencoder.loss(z, truth).item()
                        # counterfactual error as small as possible
                        if anchor_label != j: 
                            rec_error *= -1
                            curr_row['rec_cf'] = rec_error
                        else:
                            curr_row['rec_f'] = rec_error
                    # measure the similarity
                    similarity = 1 / (1 + GraphEditDistanceMetric().evaluate(anchor_instance, instance))
                    curr_row[f'sim_{i}'] = similarity
                # set the class (either positive or negative)
                # and the indices of the tuples for reference purposes
                curr_row['y'] = cls
                curr_row['tuple'] = indices
                rows.append(curr_row)
                
        return rows
            
    def transform_data(self, oracle: Oracle, dataset: Dataset, index_label=False, use_oracle=True) -> List[GeometricDataLoader]:
        indices = dataset.get_split_indices()[self.fold_id]['train']        

        data_dict_cls = {cls:[] for cls in dataset.get_classes()}
        for i in indices:
            # get instance from the dataset
            instance = dataset.get_instance(i)
            # if we're not using the oracle, then just put 0 as a dummy key
            label = oracle.predict(instance) if use_oracle else 0                   
            data_dict_cls[label].append(self.__to_geometric(instance, label=label if not index_label else int(i)))
            
        data_loaders = []
        for cls in data_dict_cls.keys():
            data_loaders.append(GeometricDataLoader(
                TorchGeometricDataset(data_dict_cls[cls]),
                                      batch_size=1,
                                      num_workers=2)
            )
        
        return data_loaders
    
    def __contrastive_learning(self, oracle: Oracle, dataset: Dataset, indices: np.array, use_oracle=True) -> Tuple[GeometricDataLoader, GeometricDataLoader]:
        # get only the training data to avoid going out of bounds
        instances = np.array(dataset.instances, dtype=object)[indices]
        
        split_indices = {label: [] for label in range(self.num_classes)}
        for i, instance in enumerate(instances):
            if use_oracle:
                split_indices[oracle.predict(instance)].append(i)
            else:
                split_indices[instance.graph_label].append(i)
            
        class_indices = sorted(list(split_indices.values()))
        
        # transform the entire dataset into torch geometric Data objects
        data = []
        for i in indices:
            data.append(self.__to_geometric(instances[i], label=int(i)))
        # generate positive and negative samples
        combos = list(product(*class_indices))
        # filter the combinations to keep only the positive tuples
        positive_combos = list([t for t in combos if len(set(t)) == len(t)])
        positive_combos = set(self.__get_tuple_permutations(positive_combos))
        
        # initialise the list of negative tuples
        _max = 0
        for indices in class_indices:
            _max = max(_max, max(indices))
            
        negative_combos = set(combinations(range(_max + 1), len(class_indices)))
        negative_combos = negative_combos.difference(positive_combos)
        
        positive_combos = list(map(list, positive_combos))
        negative_combos = list(map(list, negative_combos))
        
        if not positive_combos or not negative_combos:
            return None, None
        
        for i, positive_index in enumerate(positive_combos):
            positive_combos[i] = list(itemgetter(*positive_index)(data))
            
        for i, negative_index in enumerate(negative_combos):
            negative_combos[i] = list(itemgetter(*negative_index)(data))
                    
        pos_data_loader = GeometricDataLoader(TorchGeometricDataset(positive_combos),
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
        
        neg_data_loader = GeometricDataLoader(TorchGeometricDataset(negative_combos),
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=2)
        
        return pos_data_loader, neg_data_loader
    
    def __to_geometric(self, instance: DataInstance, label=0) -> Data:
        adj_matrix = instance.to_numpy_array()
        features = instance.features
        return self.__build_geometric_data_instance(adj_matrix, features, label)
    
    
    def __expand(self, data: Data):
        x = data.x.squeeze(dim=0).to(self.device)
        edge_index = data.edge_index.squeeze(dim=0).to(self.device)
        edge_weight = data.edge_attr.squeeze(dim=0).to(self.device)
        label = data.y.squeeze(dim=0).to(self.device)
        
        truth = torch.zeros(size=(data.num_nodes, data.num_nodes)).double()
        truth[edge_index[0,:], edge_index[1,:]] = edge_weight
        truth[edge_index[1,:], edge_index[0,:]] = edge_weight
        
        return x, edge_index, edge_weight, label, truth
    
    def __build_geometric_data_instance(self, adj_matrix, features, label):
        adj = torch.from_numpy(adj_matrix).double()
        x = torch.from_numpy(features).double()
        w = adj

        a = torch.nonzero(adj)
        w = w[a[:,0], a[:,1]]
        
        return Data(x=x, y=label, edge_index=a.T, edge_attr=w)
    
    def __get_tuple_permutations(self, tuples_list):
        permutations_list = []
        
        for tup in tuples_list:
            tup_permutations = list(permutations(tup))
            permutations_list.extend(tup_permutations)
        
        return permutations_list
    
    
    def __train(self, data_loader: GeometricDataLoader, cls: int):
        optimiser = torch.optim.Adam(self.autoencoders[cls].parameters(), lr=self.lr)
            
        for epoch in range(self.epochs_ae):
            
            losses = []
            for item in data_loader:
                x, edge_index, edge_weight, _, truth = self.__expand(item)

                optimiser.zero_grad()

                z = self.autoencoders[cls].encode(x, edge_index=edge_index, edge_weight=edge_weight)
                loss = self.autoencoders[cls].loss(z, truth)
                                                
                loss.backward()
                optimiser.step()
                                    
                losses.append(loss.item())
            
            print(f'Class {cls}, Epoch = {epoch} ----> Loss = {np.mean(losses)}')
        
        self.save_autoencoder(self.autoencoders[cls], cls)