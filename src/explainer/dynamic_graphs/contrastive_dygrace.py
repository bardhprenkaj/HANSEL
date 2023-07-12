
import os
from typing import List

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.dataset.torch_geometric.dataset_geometric import GraphPairDataset
from src.explainer.dynamic_graphs.contrastive_models.factory import AEFactory
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class ContrastiveDyGRACE(Explainer):
    
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
                 autoencoder_name='gae_contrastive',
                 top_k_cf=5,
                 config_dict=None,
                 **kwargs) -> None:
        
        super().__init__(id, config_dict)
        
        self.num_classes = num_classes
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.epochs_ae = epochs_ae
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.optimizers = [
            torch.optim.Adam(self.autoencoders[cls].parameters(), lr=self.lr) for cls in range(num_classes)
        ]
        
        self.explainer_store_path = explainer_store_path        
        self.K = top_k_cf

        
        
    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        explainer_name = f'{self.__class__.__name__}_fit_on_{dataset.name}_fold_id_{self.fold_id}'
        self.explainer_uri = os.path.join(self.explainer_store_path, explainer_name)
        self.name = explainer_name
        # train using the oracle only in the first iteration
        if self.iteration == 0:        
            self.fit(oracle, dataset, self.fold_id)
        ########################################
        # inference
        best_cf_ids, factual_ae_index = self.inference(instance, dataset)
        counterfactuals = [dataset.get_instance(id) for id in best_cf_ids]
        # update the factual autoencoder for the next iterations
        if self.iteration > 0:
            self.update(instance, counterfactuals, factual_ae_index)
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
    
    def update(self, anchor_instance: DataInstance, counterfactuals: List[DataInstance], factual_ae_index: int = 0):
        # build a dummy dataset where the factual autoencoder can update on
        curr_batch_size = self.K // 2
        # we don't need to update when there are not enough counterfactuals
        if len(counterfactuals) >= curr_batch_size:
            dataset_to_update_on = GraphPairDataset([anchor_instance], counterfactuals, oracle=None)
            data_loader = GeometricDataLoader(dataset_to_update_on, batch_size=curr_batch_size,
                                              drop_last=True, shuffle=True, follow_batch=['x_s','x_t'])
            # update the factual autoencoder on this dummy dataset
            self.__train(data_loader, cls=factual_ae_index)
               
    def __fit(self, oracle: Oracle, dataset: Dataset):
        for cls in dataset.get_classes():
            data_loader = self.transform_data(oracle, dataset, cls=cls)
            self.__train(data_loader, cls=cls)
                    
    def __rebuild_truth(self, num_nodes, edge_indices, edge_weight):
        truth = torch.zeros(size=(num_nodes, num_nodes)).double()
        truth[edge_indices[0,:], edge_indices[1,:]] = edge_weight
        truth[edge_indices[1,:], edge_indices[0,:]] = edge_weight
        return truth
    
    def __rebuild_labels(self, labels: Tensor, node_indices: Tensor) -> Tensor:
        # create a new labels tensor
        new_labels = torch.zeros(size=(node_indices[-1], node_indices[-1]))
        # set the new labels
        from_index = 0
        for i in range(1, len(node_indices)):
            # find how many nodes are there for each graph in the batch
            to_index = node_indices[i] - node_indices[i-1]
            # set the labels
            new_labels[from_index:to_index, :] = labels[i-1]
            new_labels[:, from_index:to_index] = labels[i-1]
            # swap the indices
            from_index = to_index
        return new_labels    
    
    def inference(self, search_instance: DataInstance, dataset: Dataset):
        indices = dataset.get_split_indices()[self.fold_id]['train']
        targets = [dataset.get_instance(i) for i in indices]
        
        dataset = GraphPairDataset([search_instance], targets, oracle=None)
        data_loader = GeometricDataLoader(dataset, batch_size=1, shuffle=True)
        
        anchor = dataset.to_geometric(search_instance)
        factual_ae_index = self.__get_anchor_autoencoder(anchor)
        
        with torch.no_grad():
            contrastive_losses = {}
            for batch in data_loader:
                graph1_x = batch.x_s.to(self.device)
                graph1_edge_index = batch.edge_index_s.to(self.device)
                graph1_edge_attr = batch.edge_attr_s.to(self.device)
                                    
                graph2_x = batch.x_t.to(self.device)
                graph2_edge_index = batch.edge_index_t.to(self.device)
                graph2_edge_attr = batch.edge_attr_t.to(self.device)
                
                cf_id = batch.index_t

                label = batch.label.to(self.device)
                
                z1 = self.autoencoders[factual_ae_index].encode(graph1_x, edge_index=graph1_edge_index, edge_weight=graph1_edge_attr)
                z2 = self.autoencoders[factual_ae_index].encode(graph2_x, edge_index=graph2_edge_index, edge_weight=graph2_edge_attr)
                
                # we don't need the reconstruction error here
                # since the contrastive loss gives us the necessary information
                # on what are the possible valid counterfactuals
                graph1_truth = self.__rebuild_truth(num_nodes=batch.x_s.shape[0], edge_indices=batch.edge_index_s, edge_weight=batch.edge_attr_s)
                _, contrastive_loss = self.autoencoders[factual_ae_index].loss([z1,z2], [graph1_truth, label])

                contrastive_losses[contrastive_loss.item()] = cf_id.item()
        # get the lowest k contrastive losses
        best_cf = sorted(list(contrastive_losses.keys()))
        best_cf = best_cf[:self.K]
        return [contrastive_losses[id] for id in best_cf], factual_ae_index
        
        
    def transform_data(self, oracle: Oracle, dataset: Dataset, cls=0) -> GeometricDataLoader:
        indices = dataset.get_split_indices()[self.fold_id]['train']
        
        anchors, targets = [], []
        for i in indices:
            inst = dataset.get_instance(i)
            if oracle.predict(inst) == cls:
                anchors.append(inst)
            else:
                targets.append(inst)
            
        # create dataset of positive pairs    
        positive_dataset = GraphPairDataset(anchors, anchors, oracle)
        # create dataset of negative pairs
        negative_dataset = GraphPairDataset(anchors, targets, oracle)
        # concatenate positive and negative pairs
        dataset_of_pairs = torch.utils.data.ConcatDataset([positive_dataset, negative_dataset])
        # data loader with batches
        return GeometricDataLoader(dataset_of_pairs, batch_size=self.batch_size,
                                   drop_last=True, shuffle=True, follow_batch=['x_s', 'x_t'])
    
    def __get_anchor_autoencoder(self, anchor: Data) -> int:
        with torch.no_grad():
            x = anchor.x.to(self.device)
            edge_index = anchor.edge_index.to(self.device)
            edge_weights = anchor.edge_attr.to(self.device)
            truth = self.__rebuild_truth(x.shape[0], edge_index, edge_weights)
            
            rec_errors = []
            for autoencoder in self.autoencoders:
                z = autoencoder.encode(x, edge_index, edge_weights)
                # we don't care about the contrastive loss here
                # we just need to take the reconstruction error
                zs = [z,z]
                # the second element of the truth "tensor" can be anything
                # since we don't care about the contrastive loss
                rec_error, _ = autoencoder.loss(zs, [truth, 42])
                rec_errors.append(rec_error)
                            
            return np.argmin(rec_errors)
        
        
    def __train(self, data_loader: GeometricDataLoader, cls: int = 0):
        for epoch in range(self.epochs_ae):
            rec_losses = []
            contrastive_losses = []
            for batch in data_loader:
                graph1_x = batch.x_s.to(self.device)
                graph1_edge_index = batch.edge_index_s.to(self.device)
                graph1_edge_attr = batch.edge_attr_s.to(self.device)
                                    
                graph2_x = batch.x_t.to(self.device)
                graph2_edge_index = batch.edge_index_t.to(self.device)
                graph2_edge_attr = batch.edge_attr_t.to(self.device)
                
                label = batch.label.to(self.device)
                
                self.optimizers[cls].zero_grad()
                
                z1 = self.autoencoders[cls].encode(graph1_x, edge_index=graph1_edge_index, edge_weight=graph1_edge_attr)
                z2 = self.autoencoders[cls].encode(graph2_x, edge_index=graph2_edge_index, edge_weight=graph2_edge_attr)
                
                # rebuild the ground truth and the labels to calculate the contrastive loss
                graph1_truth = self.__rebuild_truth(num_nodes=batch.x_s.shape[0], edge_indices=batch.edge_index_s, edge_weight=batch.edge_attr_s)
                labels = self.__rebuild_labels(label, batch.x_s_ptr)
                
                rec_loss, contrastive_loss = self.autoencoders[cls].loss([z1, z2], [graph1_truth, labels])
                
                (rec_loss + contrastive_loss).backward()
                self.optimizers[cls].step()
                
                rec_losses.append(rec_loss.item())
                contrastive_losses.append(contrastive_loss.item())
                
            print(f'Class {cls}, Epoch = {epoch} ----> Rec loss = {np.mean(rec_losses)}, Contrastive loss = {np.mean(contrastive_losses)}')


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