import math
import os

import networkx as nx
import numpy as np
import torch
from dgl import from_networkx, to_networkx
from torch.utils.data import Dataset

from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle
from src.oracle.oracle_cf2 import CustomDGLDataset


class CF2Explainer(Explainer):
    def __init__(
        self,
        id,
        explainer_store_path,
        n_nodes,
        converter,
        batch_size_ratio=0.1,
        lr=1e-3,
        weight_decay=0,
        gamma=1e-4,
        lam=1e-4,
        alpha=1e-4,
        epochs=200,
        fold_id=0,
        config_dict=None,
    ) -> None:

        super().__init__(id, config_dict)

        self.name = "cf2"
        self.n_nodes = n_nodes
        self.converter = converter
        self.batch_size_ratio = batch_size_ratio
        self.fold_id = fold_id
        self.explainer_store_path = explainer_store_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.epochs = epochs
        self.fold_id = fold_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._fitted = False
        
        self.explainer = ExplainModelGraph(self.n_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.explainer.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        
    def explain(self, instance: DataInstance, oracle: Oracle, dataset: Dataset):
        if(not self._fitted):
            self.fit(dataset, oracle)

        self.explainer.eval()
        new_dataset = Dataset(id='dummy')
        new_dataset.instances.append(instance)
        new_dataset = self.converter.convert(new_dataset)
        features_and_weight_instance = new_dataset.instances[-1]
        
        try:
            with torch.no_grad():
                adj = features_and_weight_instance.to_numpy_array()
                weights = features_and_weight_instance.weights
                features = features_and_weight_instance.features
                
                g = from_networkx(nx.from_numpy_array(adj))
                # set node features of the graph
                g.ndata['feat'] = torch.from_numpy(features).float()
                # set the edge weights of the graph
                g.edata['weights'] = torch.from_numpy(weights).float()
                
                weighted_adj = self.explainer._rebuild_weighted_adj(g)
                masked_adj = self.explainer.get_masked_adj(weighted_adj).numpy()

                cf_instance = DataInstanceWFeaturesAndWeights(instance.id)
                cf_instance.from_numpy_array(masked_adj)
                cf_instance = self.converter.convert_instance(cf_instance)    
                        
                print(f'Finished evaluating for instance {instance.id}')
                return [cf_instance]
        except:
            return [instance]
        

    def save_explainers(self):
        torch.save(
            self.explainer.state_dict(),
            os.path.join(self.explainer_store_path, self.name, f"explainer"),
        )

    def load_explainers(self):
        self.explainer.load_state_dict(
            torch.load(os.path.join(self.explainer_store_path, self.name, f"explainer"))
        )

    def fit(self, dataset: Dataset, oracle: Oracle):
        explainer_name = (
            f"cf2_fit_on_{dataset.name}_fold_id={self.fold_id}"
        )
        explainer_uri = os.path.join(self.explainer_store_path, explainer_name)
        self.name = explainer_name

        if os.path.exists(explainer_uri):
            self.load_explainers()
        else:
            os.mkdir(explainer_uri)
            self.__fit(dataset, oracle)
            self.save_explainers()
        self._fitted = True
        
    def __fit(self, dataset: Dataset, oracle: Oracle):
        if self.converter:
            dataset = self.converter.convert(dataset)
        graphs, labels = self.transform_data(dataset)
        
        self.explainer.train()
        for epoch in range(self.epochs):
            
            losses = list()
            
            for i, graph in enumerate(graphs):
                graph = graph.to(self.device)

                pred1, pred2 = self.explainer(graph, oracle, label=labels[i])
                
                loss = self.explainer.loss(graph,
                                           pred1, pred2,
                                           self.gamma, self.lam,
                                           self.alpha)
                
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1} --- loss {np.mean(losses)}")
            
        self._fitted = True

    def transform_data(self, dataset: Dataset):
        adj  = np.array([i.to_numpy_array() for i in dataset.instances], dtype=object)
        features = np.array([i.features for i in dataset.instances], dtype=object)
        weights = np.array([i.weights for i in dataset.instances], dtype=object)
        y = np.array([i.graph_label for i in dataset.instances], dtype=np.float32)[..., np.newaxis]
        """# adj.shape = n x n
        adj = self.__pad(adj)
        # feature.shape = n x n
        features = self.__pad(features)
        # weights.shape = n x n
        weights = self.__pad(weights, is_adj=False)"""

        indices = dataset.get_split_indices()[self.fold_id]['train'] 
        adj, features, weights, y = adj[indices], features[indices], weights[indices], y[indices]
        dgl_dataset = CustomDGLDataset(adj, features, weights, y)
        
        return dgl_dataset.graphs, dgl_dataset.labels
    
    def __pad(self, arr, is_adj=True):
        # pad the array to the highest number of nodes
        # Find the maximum number of columns (second dimension) among all matrices
        max_cols = max(len(matrix) for matrix in arr)
        # Initialize a new tensor to store the padded matrices
        if is_adj:
            padded_tensor = np.zeros((len(arr), max_cols, max_cols))
        else:
            padded_tensor = np.zeros((len(arr), max_cols, arr.shape[-1]))
        # Pad each matrix in the tensor
        for i in range(len(arr)):
            rows, cols = arr[i].shape
            padded_tensor[i, :rows, :cols] = arr[i]
        padded_tensor = padded_tensor.astype(np.float64)
        return padded_tensor

class ExplainModelGraph(torch.nn.Module):
    
    def __init__(self, n_nodes: int):
        super(ExplainModelGraph, self).__init__()

        self.n_nodes = n_nodes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mask = self.build_adj_mask()

    def forward(self, graph, oracle, label=None, index=None):        
        adjacency = nx.to_numpy_array(to_networkx(graph))
        weights = graph.edata['weights'].detach().numpy()
        features = graph.ndata['feat'].detach().numpy()

        instance = DataInstanceWFeaturesAndWeights(-1)
        instance.from_numpy_array(adjacency)
        instance.weights = weights
        instance.features = features
        instance.graph_label = label 
        pred1 = oracle.predict(instance)

        # re-build weighted adjacency matrix
        weighted_adj = self._rebuild_weighted_adj(graph)
        # get the masked_adj
        masked_adj = self.get_masked_adj(weighted_adj)
        # get the new weights as the difference between
        # the weighted adjacency matrix and the masked learned
        new_weights = weighted_adj - masked_adj
        # get only the edges that exist
        row_indices, col_indices = torch.where(new_weights != 0)

        cf_instance = DataInstanceWFeaturesAndWeights(instance.id)
        cf_instance.from_numpy_array(adjacency)
        cf_instance.features = features
        cf_instance.weights = new_weights[row_indices, col_indices].detach().numpy()
        pred2 = oracle.predict(cf_instance)

        pred1 = torch.Tensor([pred1]).float()  # factual
        pred2 = torch.Tensor([pred2]).float()  # counterfactual

        return pred1, pred2

    def build_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.n_nodes + self.n_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self, weights):
        sym_mask = torch.sigmoid(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        masked_adj = weights * sym_mask
        return masked_adj

    def loss(self, graph, pred1, pred2, gam, lam, alp):
        weights = self._rebuild_weighted_adj(graph)
        bpr1 = torch.nn.functional.relu(gam + 0.5 - pred1)  # factual
        bpr2 = torch.nn.functional.relu(gam + pred2 - 0.5)  # counterfactual
        masked_adj = torch.flatten(self.get_masked_adj(weights))
        L1 = torch.linalg.norm(masked_adj, ord=1)
        return L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
    
    
    def _rebuild_weighted_adj(self, graph):
        u,v = graph.all_edges(order='eid')
        weights = np.zeros((self.n_nodes, self.n_nodes))
        weights[u.numpy(), v.numpy()] = graph.edata['weights'].detach().numpy()
        return torch.from_numpy(weights).float()