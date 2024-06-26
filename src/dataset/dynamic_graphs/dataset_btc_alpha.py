from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset

import networkx as nx
import numpy as np
import os
import random
import pandas as pd
import itertools
from typing import Dict, List



class BTCAlpha(DynamicDataset):
    
    def __init__(self,
                 id,
                 begin_t,
                 end_t,
                 filter_min_graphs=10,
                 number_of_communities=15,
                 padding=True,
                 config_dict=None) -> None:
        
        super().__init__(id, begin_t, end_t, config_dict)
        self.name = 'btc_alpha'   
        self.filter_min_graphs = filter_min_graphs
        self.number_of_communities = number_of_communities
        self.padding = padding         
    
    def read_csv_file(self, dataset_path):
        # read the number of vertices in for each simplex
        ratings = pd.read_csv(os.path.join(dataset_path, 'network.txt'), sep='\t')
        ratings.columns=['source','target','time','rating','comment']
        ratings['year'] = ratings.time.apply(lambda x : pd.to_datetime(x).year)
        # retain only desired history
        ratings = ratings[(ratings.year >= self.begin_t) & (ratings.year <= self.end_t)]
        print("Now proceeding to grouping")
        self.grouped_by_time = ratings.groupby('year')
        
    
    def build_temporal_graph(self):
        self.unprocessed_data = {}
        self.mean_weights = 0
        it = 0
        for year, df in self.grouped_by_time:
            print(f'Working for time={year}')
            G = nx.DiGraph()
            source, target, weights = df.source.values.tolist(), df.target.values.tolist(), df.rating.values.tolist()
            if it == 0:
                self.mean_weights = np.mean(weights)
            G.add_nodes_from(source + target)
            for u, v, w in zip(source, target, weights):
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=w)
                
            # year -> entire graph of coauthor simplices
            self.unprocessed_data[year] = G
            it += 1
            
        self.preprocess_datasets()
                
                
    def preprocess_datasets(self):
        print("Preprocessing began...")
        self.__get_communities(self.unprocessed_data)
        print('Eliminating the empty snapshots')
        # clear those snapshots that are empty
        for i in range(self.begin_t, self.end_t + 1):
            self.dynamic_graph[i].name = str(i)
            if self.dynamic_graph[i].get_data_len() <= self.number_of_communities:
                self.dynamic_graph.pop(i, None)
        print(self.dynamic_graph)
        print(f'Finished preprocessing.')
                
    def __get_communities(self, temporal_graph):
        for t in range(max(self.begin_t, min(temporal_graph.keys())), min(self.end_t, max(temporal_graph.keys()))+1):
            instance_id = 0
            nodes = list(temporal_graph[t].nodes)
            for node in nodes[:int(len(nodes) * .2)]:
                subgraph = nx.ego_graph(temporal_graph[t], node, undirected=True)
                self.__create_data_instance(id=instance_id, graph=subgraph,
                                            year=t, label=self.__get_label(subgraph))
                instance_id += 1
                
        if self.padding:
            self.__pad()
            
            
    def __pad(self):
        arrays = []
        num_instances = {}
        for year in self.dynamic_graph.keys():
            num_instances[year] = len(self.dynamic_graph[year].instances)
            for inst in self.dynamic_graph[year].instances:
                arrays.append(inst.to_numpy_array(store=False))
                
        del num_instances[min(self.dynamic_graph.keys())]
        max_dimension = max([arr.shape[0] for arr in arrays])
        
        # Pad each array to the highest dimension
        padded_matrices = [np.pad(matrix, 
                                  [(0, max_dimension - matrix.shape[0]), 
                                   (0, max_dimension - matrix.shape[1])] if matrix.ndim == 2\
                                       else [(0, max_dimension - matrix.shape[0])], mode='constant', constant_values=0)\
                                           for matrix in arrays]
        for key_num, year in enumerate(self.dynamic_graph.keys()):
            for i, inst in enumerate(self.dynamic_graph[year].instances):
                j = i + num_instances[year] if key_num > 0 else i
                inst.from_numpy_array(padded_matrices[j], store=True)
                self.dynamic_graph[year].instances[i] = inst   
    
    def __create_data_instance(self, id: int, graph: nx.Graph, year: int, label: float):
        instance = DataInstance(id=id)
        instance.name = f'rating_graph={id}'
        instance.graph = graph
        instance.graph_label = label
        instance = self.__generate_node_features(instance)
        print(f'Adding DataInstance with id = {id} @year={year} with label = {label}')
        self.dynamic_graph[year].instances.append(instance)
    
    def __get_label(self, graph: nx.Graph) -> Dict[int, int]:
        ratings = np.array(list(nx.get_edge_attributes(graph, 'weight').values()))
        return 1 if np.sum(ratings < 0) > np.log(np.sum(ratings >= 0)) else 0      

    def __generate_node_features(self, instance: DataInstance) -> DataInstanceWFeaturesAndWeights:
        graph = instance.graph
        # Calculate the betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(graph)
        # Calculate the closeness centrality
        closeness_centrality = nx.closeness_centrality(graph)
        # Calculate the harmonic centrality
        harmonic_centrality = nx.harmonic_centrality(graph)
        # Calculate the clustering coefficient
        clustering_coefficient = nx.clustering(graph)
        # stack the above calculations and transpose the matrix
        # the new dimensionality is num_nodes x 4
        features = np.stack((list(betweenness_centrality.values()),
                             list(closeness_centrality.values()),
                             list(harmonic_centrality.values()),
                             list(clustering_coefficient.values())), axis=0).T
        # copy the instance information and set the node features
        new_instance = DataInstanceWFeaturesAndWeights(id=instance.id)
        new_instance.from_numpy_matrix(nx.adjacency_matrix(graph))
        new_instance.features = features
        new_instance.graph_label = instance.graph_label
        new_instance.graph_dgl = instance.graph_dgl
        new_instance.edge_labels = instance.edge_labels
        new_instance.node_labels = instance.node_labels
        new_instance.name = instance.name
        new_instance.weights = nx.to_numpy_array(graph)
        # return the new instance with features
        return new_instance
    
        
        