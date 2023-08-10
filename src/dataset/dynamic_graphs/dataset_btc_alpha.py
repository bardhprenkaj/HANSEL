from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeatures, DataInstanceWFeaturesAndWeights
from src.dataset.dataset_base import Dataset
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset


import networkx as nx
import numpy as np
import os
import pandas as pd
import community
from typing import Dict, List

from collections import defaultdict

class BTCAlpha(DynamicDataset):
    
    def __init__(self,
                 id,
                 begin_t,
                 end_t,
                 filter_min_graphs=10,
                 config_dict=None) -> None:
        
        super().__init__(id, begin_t, end_t, config_dict)
        self.name = 'btc_alpha'   
        self.filter_min_graphs = filter_min_graphs         
        
    def read_csv_file(self, dataset_path):
        # read the number of vertices in for each simplex
        ratings = pd.read_csv(os.path.join(dataset_path, 'soc-sign-bitcoinalpha.edges'), header=None, names=['source','target','rating','time'])
        ratings.time = ratings.time.apply(lambda x : pd.to_datetime(x, unit='s'))
        ratings['year'] = ratings.time.apply(lambda x : x.year)
        ratings = ratings.sort_values(by='year')
        # retain only desired history
        ratings = ratings[(ratings.year >= self.begin_t) & (ratings.year <= self.end_t)]
        
        self.grouped_by_time = ratings.groupby('year')
        
    
    def build_temporal_graph(self):
        self.unprocessed_data = {}
        for year, df in self.grouped_by_time:
            print(f'Working for time={year}')
            G = nx.Graph()
            source, target, weights = df.source.values.tolist(), df.target.values.tolist(), df.rating.values.tolist()
            G.add_nodes_from(source + target)
            for u, v, w in zip(source, target, weights):
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=w)
            # year -> entire graph of coauthor simplices
            self.unprocessed_data[year] = G
        
        self.preprocess_datasets()
                
                
    def preprocess_datasets(self):
        print("Preprocessing began...")
        self.__get_communities(self.unprocessed_data)
        print('Eliminating the empty snapshots')
        # clear those snapshots that are empty
        for i in range(self.begin_t, self.end_t + 1):
            if self.dynamic_graph[i].get_data_len() > 0:
                self.dynamic_graph.pop(i, None)
        print(f'Finished preprocessing.')
                
    def __get_communities(self, temporal_graph):
        for t in range(max(self.begin_t, min(temporal_graph.keys())), min(self.end_t, max(temporal_graph.keys()))+1):
            instance_id = 0
            communities = nx.community.girvan_newman(temporal_graph[t])
            for community in communities:
                for nodes in community:
                    subgraph = temporal_graph[t].subgraph(list(nodes))
                    if subgraph.number_of_nodes() > self.filter_min_graphs:
                        print(subgraph)
                        self.__create_data_instance(id=instance_id, graph=subgraph,
                                                    year=t, label=self.__get_label(subgraph))
                        instance_id += 1                      
    
    def __create_data_instance(self, id: int, graph: nx.Graph, year: int, label: float):
        instance = DataInstance(id=id)
        instance.name = f'rating_graph={id}'
        instance.graph = graph
        instance.graph_label = label
        instance = self.__generate_node_features(instance)
        print(f'Adding DataInstance with id = {id} @year={year}')
        self.dynamic_graph[year].instances.append(instance)
    
    def __get_label(self, graph: nx.Graph) -> Dict[int, int]:      
        ratings = np.array(list(nx.get_edge_attributes(graph, 'weight').values()))
        return 1 if np.sum(ratings > 0) >= np.sum(ratings < 0) else 0            

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
    
        
        