
import json
import os
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import TorchGeometricDataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle
from src.utils.weight_schedulers import WeightScheduler


class ConDGCE(Explainer):
    
    def __init__(self,
                 id: int,
                 explainer_store_path: str,
                 autoencoders: List[torch.nn.Module],
                 alpha_scheduler: WeightScheduler,
                 beta_scheduler: WeightScheduler,
                 fold_id: int = 0,
                 batch_size: int = 24,
                 lr: float = 1e-4,
                 epochs: int = 100,
                 lam: int = 0.5,
                 k: int = 5,
                 wandb_optimize=False,
                 config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.explainer_store_path = explainer_store_path        
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.autoencoders = autoencoders
        self.k = k
        self.lam = lam
        self.alpha_scheduler, self.beta_scheduler = alpha_scheduler, beta_scheduler
        self.optimizers = [
            torch.optim.Adam(autoencoder.parameters(), lr=self.lr) for autoencoder in self.autoencoders
        ]
        self.iteration = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wandb_optimize = wandb_optimize
        
       
        self.prior_y_prob = {str(cls) : 0 for cls in range(len(self.autoencoders))}
        self.next_iteration_classifications = list()
        
        self.counterfactuals = {cls : [] for cls in range(len(self.autoencoders))}

        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        explainer_name = f'{self.__class__.__name__}_fit_on_{dataset.name}_fold_id_{self.fold_id}'\
            + f'_batch={self.batch_size}_lr={self.lr}_e={self.epochs}_k={self.k}'\
                    + f'_alpha={self.alpha_scheduler.__class__.__name__}'\
                        + f'_beta={self.beta_scheduler.__class__.__name__}'
                        
        self.explainer_uri = os.path.join(self.explainer_store_path, explainer_name)
        self.name = explainer_name
        # train using the oracle only in the first iteration
        if self.iteration == 0:        
            self.fit(oracle, dataset, self.fold_id)
            self.prev_iteration = self.iteration
        ########################################
        # inference
        counterfactuals, factual_index = self.inference(instance, dataset)
        # update the factual autoencoder for the next iterations
        if self.iteration > 0:
            self.reflect_changes(instance, counterfactuals, factual_index)
        ########################################
        # save the autoencoders
        # we do the saving here since in the next iterations
        # one of the autoencoders might have never been used
        # we need to "propagate" the weights of the
        # currently-not-trained autoencoder to the next iteration
        for cls, autoencoder in enumerate(self.autoencoders):
            self.save_autoencoder(autoencoder, cls)
        #######################################
        # return the counterfactual list
        return counterfactuals
        
    def fit(self, oracle: Oracle, dataset: Dataset, fold_id=0):
        if os.path.exists(self.explainer_uri):
            # Load the weights of the trained model
            self.load_autoencoders()
        else:
            self.__fit(oracle, dataset)
    
    def reflect_changes(self, search_instance: DataInstance, counterfactuals: List[DataInstance], factual_index: int = 0):
        factual_loader = DataLoader(TorchGeometricDataset([search_instance]))
        cf_loader = DataLoader(TorchGeometricDataset(counterfactuals))
        data_loaders = (cf_loader, factual_loader) if factual_index else (factual_loader, cf_loader)
        # save the classification labels
        self.next_iteration_classifications.append(factual_index)
        self.next_iteration_classifications += [1-factual_index] * len(counterfactuals)
        # update both autoencoders
        for cls in range(len(self.autoencoders)):
            epochs = max(self.epochs // 10, 1) if cls == factual_index else len(counterfactuals)
            self.__train(cls, data_loaders[cls], epochs)
               
    def __fit(self, oracle: Oracle, dataset: Dataset):
        data_loaders: Tuple[DataLoader, DataLoader] = self.transform_data(oracle, dataset)
        for cls in range(len(self.autoencoders)):
            self.__train(cls, data_loaders[cls], self.epochs)
                    
    @torch.no_grad()
    def inference(self, search_instance: DataInstance, dataset: Dataset):
        # get the instances of this current snapshot
        indices = dataset.get_split_indices()[self.fold_id]['train']
        instances = [dataset.get_instance(i) for i in indices]
        # transform these instances into a geometric dataset
        new_dataset = TorchGeometricDataset(instances)
        # this is the dataset corresponding to the anchor point
        # that we want to find counterfactuals for
        search_graph = TorchGeometricDataset([search_instance]).get(0)
        # predict the class in a generative fashion
        factual_index = self.generative_classification(search_graph)
        # get the opposite class as the plausible counterfactual one
        cf_index = 1 - factual_index
        # get the autoencoder corresponding to the counterfactuals
        cf_vgae = self.autoencoders[cf_index]
        # embed this graph to its plausible representation
        # that could lie outside the representation region
        # of the data this autoencoder has been trained on
        z = self.embed(cf_vgae, search_graph)
        # pull this representation towards the center 
        # of the "known world" for this autoencoder
        z_star = self.lam * z
        # get the nearest neighbours
        cf_indices = self.neighbours(cf_vgae, z_star, new_dataset, cf_index)
        # return the counterfactuals
        counterfactuals = np.array(instances)[cf_indices]
        self.counterfactuals[cf_index].append(counterfactuals.tolist())
        return counterfactuals, factual_index
    
    @torch.no_grad()
    def neighbours(self, autoencoder: torch.nn.Module, latent: Tensor, dataset: TorchGeometricDataset, cls=0):
        embeddings = []
        for inst in dataset.instances:
            if inst not in self.counterfactuals[1-cls]:
                embeddings.append(self.embed(autoencoder, inst).numpy())
        embeddings = torch.from_numpy(np.array(embeddings))
        # calculate the pairwise Euclidean distance between latent and embeddings
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        latent = latent.reshape(1, -1)
        squared_distance = torch.sum((latent -  embeddings) ** 2, dim=1)
        # Take the square root to get the Euclidean distance
        distance = torch.sqrt(squared_distance)
        _, top_k_indices = torch.topk(distance, min(self.k, dataset.len()), dim=0, largest=False)
        return top_k_indices.numpy()
        
    def transform_data(self, oracle: Oracle, dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        indices = dataset.get_split_indices()[self.fold_id]['train']
        
        separated_data = {cls: [] for cls in range(len(self.autoencoders))}
        for i in indices:
            inst = dataset.get_instance(i)
            separated_data[oracle.predict(inst)].append(inst)
        # we need this probability of choosing which autoencoder
        # to use at inference time for generating counterfactuals
        for cls in separated_data.keys():
            self.prior_y_prob[str(cls)] = len(separated_data[cls]) / len(indices)
        print(self.prior_y_prob)
        # create dataset of positive pairs    
        positive_dataset = TorchGeometricDataset(separated_data[0])
        # create dataset of negative pairs
        negative_dataset = TorchGeometricDataset(separated_data[1])
        # data loader with batches
        return DataLoader(positive_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True),\
            DataLoader(negative_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True)
    
    @torch.no_grad()
    def generative_classification(self, anchor: Data) -> int:
        """
        Get the index of the anchor autoencoder with the
        minimum reconstruction error for a given anchor data.

        Args:
            anchor (Data): The anchor data containing node features,
                           edge indices, and edge weights.

        Returns:
            int: The index of the anchor autoencoder with the
                 minimum reconstruction error.

        Raises:
            None
        """
        class_choice = []
        for i, autoencoder in enumerate(self.autoencoders):
            z = self.embed(autoencoder, anchor)
            # get the reconstruction loss
            rec_loss = autoencoder.loss(z, anchor.edge_index)
            # get the distance loss from the center of the latent space
            dist_loss = torch.linalg.vector_norm(z, ord=2)
            loss = .5*(rec_loss + dist_loss) - np.log(self.prior_y_prob[str(i)])
            class_choice.append(loss)

        print(class_choice)
        return np.argmin(class_choice)
    
    @torch.no_grad()
    def embed(self, autoencoder: torch.nn.Module, data: Data):
        x = data.x.to(self.device)
        edge_indices = data.edge_index.to(self.device)
        edge_attrs = data.edge_attr.to(self.device)
        return autoencoder.encode(x, edge_indices, edge_attrs)
        
        
    def __train(self, cls, data_loader: DataLoader, epochs: int):
        if data_loader: # if there are any samples in this data loader
            mse_loss = torch.nn.MSELoss()
            for epoch in range(epochs):
                # loop through the batches
                feature_rec_losses = []
                dist_losses = [] 
                mse_losses = []
                for batch in data_loader:
                    x = batch.x.to(self.device)
                    edge_indices = batch.edge_index.to(self.device)
                    edge_attrs = batch.edge_attr.to(self.device)
                                    
                    self.optimizers[cls].zero_grad()
                    # get the latent representation of the graph
                    _, z = self.autoencoders[cls](x, edge_indices, edge_attrs)
                    adj_hat = self.autoencoders[cls].decoder.forward_all(z, **{'edge_index': edge_indices, 'edge_attr': edge_attrs, 'sigmoid': True})
                    # rebuild the ground truth
                    gt = self.__rebuild_adj_matrix(len(x), edge_indices, edge_attrs)
                    # get the feature reconstruction loss
                    feature_rec_loss = self.autoencoders[cls].loss(z, edge_indices, edge_attr=edge_attrs)
                    # get the edge reconstruction loss
                    edge_rec_loss = mse_loss(adj_hat, gt)
                    # make the latent representation be centred
                    dist_loss = torch.linalg.vector_norm(z, ord=2)
            
                    # minimize both losses
                    loss = feature_rec_loss + dist_loss + edge_rec_loss
                    loss.backward()
                    self.optimizers[cls].step()
                    
                    mse_losses.append(edge_rec_loss.item())
                    feature_rec_losses.append(feature_rec_loss.item())
                    dist_losses.append(dist_loss.item())

                print(f'Class {cls}, Epoch = {epoch} ----> Feature rec loss = {np.mean(feature_rec_losses): .4f}, Dist loss = {np.mean(dist_losses)}, Edge rec loss = {np.mean(mse_losses)}')
            """if self.wandb_optimize:
                wandb.log({
                    f'rec_loss_{cls}_{self.fold_id}': np.mean(rec_losses[epoch]),
                    f'contrastive_loss_{cls}_{self.fold_id}': np.mean(contrastive_losses),
                    f'epoch_{cls}_{self.fold_id}': epoch,
                })"""    
    
    def save_autoencoder(self, model: torch.nn.Module, cls: int):
        """
        Save the state_dict of an autoencoder model.

        Args:
            model (torch.nn.Module): The autoencoder model to be saved.
            cls (int): The class identifier for the autoencoder.

        Returns:
            None

        Raises:
            None
        """
        try:
            os.mkdir(self.explainer_uri)                    
        except FileExistsError:
            pass
            
        torch.save(model.state_dict(),
                 os.path.join(self.explainer_store_path, self.name, f'autoencoder_{cls}'))
        
        with open(os.path.join(self.explainer_store_path, self.name, 'priors_y.json'), 'w') as f:
            json.dump(self.prior_y_prob, f)
        
    def load_autoencoders(self):
        """
        Load the state_dicts of autoencoder models.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        for i, _ in enumerate(self.autoencoders):
            self.autoencoders[i].load_state_dict(torch.load(os.path.join(self.explainer_store_path,
                                                                         self.name, f'autoencoder_{i}')))
            
        with open(os.path.join(self.explainer_store_path, self.name, 'priors_y.json'), 'r') as f:
            self.prior_y_prob = json.load(f)
            
        
    def update(self):
        print(self.next_iteration_classifications)
        # update the priors of selecting the autoencoders
        counts = Counter(self.next_iteration_classifications)
        all_elems = sum(list(counts.values()))
        for cls in self.prior_y_prob.keys():
            self.prior_y_prob[str(cls)] = max(counts[int(cls)] / all_elems, 1e-4)
        # flush the classifications of the current iteration
        self.next_iteration_classifications.clear()


    def __rebuild_adj_matrix(self, num_nodes: int, edge_indices: Tensor, edge_weight: Tensor) -> Tensor:
        truth = torch.zeros(size=(num_nodes, num_nodes)).double()
        truth[edge_indices[0,:], edge_indices[1,:]] = edge_weight
        truth[edge_indices[1,:], edge_indices[0,:]] = edge_weight
        return truth