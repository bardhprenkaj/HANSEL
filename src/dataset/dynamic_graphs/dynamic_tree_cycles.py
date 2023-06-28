import os
from typing import Dict
from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeatures

from src.dataset.dataset_synthetic_generator import Synthetic_Data
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset
import networkx as nx

import numpy as np

class DynTreeCycles(DynamicDataset):

    def __init__(self, id, begin_t, end_t,
                 num_instances_per_snapshot=300,
                 n_nodes=300,
                 nodes_in_cycle=200,
                 config_dict=None) -> None:
        super().__init__(id, begin_t, end_t, config_dict)
        
        assert (begin_t < end_t)
        
        self._id = id
        self._name = 'dynamic_tree_cycles'

        self.begin_t = begin_t
        self.end_t = end_t
        
        self.num_instances_per_snapshot = num_instances_per_snapshot
        self.n_nodes = n_nodes
        self.nodes_in_cycle = nodes_in_cycle
        
        self.dynamic_graph: Dict[int, Synthetic_Data] = { 
                                            key: Synthetic_Data(key, config_dict=config_dict)\
                                                for key in range(self.begin_t, self.end_t + 1) 
                                            }
        
        self.splits = {
            key: None for key in range(self.begin_t, self.end_t + 1)
        }
        
        self._instance_id_counter = len(self.dynamic_graph)
        self._config_dict = config_dict
        
        
    def build_temporal_graph(self):
        for i in range(self.begin_t, self.end_t + 1):
            self.dynamic_graph[i].generate_tree_cycles_dataset(n_instances=self.num_instances_per_snapshot,
                                                               n_total=self.n_nodes, 
                                                               n_in_cycles=self.nodes_in_cycle)
            
            self.dynamic_graph[i]._name = f'{i}'
            
        self.preprocess_datasets()

    def preprocess_datasets(self):
        for i in range(self.begin_t, self.end_t + 1):
            for j, instance in enumerate(self.dynamic_graph[i].instances):
                # sort each node by its degree
                new_graph = self.__sort_nodes_by_degree(instance.graph)
                instance.from_numpy_matrix(nx.adjacency_matrix(new_graph))
                # set the node features
                self.dynamic_graph[i][j] = self.__generate_node_features(instance)
                
    def __sort_nodes_by_degree(self, original_graph: nx.Graph) -> nx.Graph:
        # Step 1: Compute the degrees of all nodes
        degrees = dict(original_graph.degree())
        # Step 2: Sort the nodes based on their degrees in descending order
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        # Step 3: Create a new empty graph with the same attributes as the original graph
        new_graph = nx.Graph(original_graph.graph)
        # Step 4: Iterate over the sorted nodes and add them to the new graph
        for node in sorted_nodes:
            new_graph.add_node(node)
        # Step 5: Add the edges and edge weights from the original graph to the new graph
        for u, v, data in original_graph.edges(data=True):
            if u in new_graph and v in new_graph:
                new_graph.add_edge(u, v, **data)
                
        return new_graph
    
    
    def __generate_node_features(self, instance: DataInstance) -> DataInstanceWFeatures:
        graph = instance.graph
        # Calculate the betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(graph)
        # Calculate the closeness centrality
        closeness_centrality = nx.closeness_centrality(graph)
        # Calculate the eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=500)
        # Calculate the harmonic centrality
        harmonic_centrality = nx.harmonic_centrality(graph)
        # Calculate the clustering coefficient
        clustering_coefficient = nx.clustering(graph)
        # stack the above calculations and transpose the matrix
        # the new dimensionality is num_nodes x 5
        features = np.stack((list(betweenness_centrality.values()),
                             list(closeness_centrality.values()),
                             list(eigenvector_centrality.values()),
                             list(harmonic_centrality.values()),
                             list(clustering_coefficient.values())), axis=0).T
        # copy the instance information and set the node features
        new_instance = DataInstanceWFeatures(id=instance.id)
        new_instance.from_numpy_matrix(nx.adjacency_matrix(graph))
        new_instance.features = features
        new_instance.graph_label = instance.graph_label
        new_instance.graph_dgl = instance.graph_dgl
        new_instance.edge_labels = instance.edge_labels
        new_instance.node_labels = instance.node_labels
        new_instance.name = instance.name
        # return the new instance with features
        return new_instance
                
        
        
