from src.dataset.data_instance_base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset

import networkx as nx
import numpy as np
import os
import pandas as pd
from typing import Dict

class Bonanza(DynamicDataset):
    
    def __init__(self,
                 id,
                 begin_t,
                 end_t,
                 filter_min_graphs=10,
                 number_of_communities=15,
                 padding=True,
                 config_dict=None) -> None:
        
        super().__init__(id, begin_t, end_t, config_dict)
        self.name = 'bonanza'   
        self.filter_min_graphs = filter_min_graphs
        self.number_of_communities = number_of_communities
        self.padding = padding         
    
    def read_csv_file(self, dataset_path):
        # read the number of vertices in for each simplex
        buyer_to_seller = pd.read_csv(os.path.join(dataset_path, 'bonanza-buyer-to-seller.csv'),
                                      engine='python',
                                      header=None,
                                      error_bad_lines=False,
                                      verbose=0,
                                      index_col=None,
                                      sep=r',|;|\t')
        
        seller_to_buyer = pd.read_csv(os.path.join(dataset_path, 'bonanza-seller-to-buyer.csv'),
                                      engine='python',
                                      header=None,
                                      error_bad_lines=False,
                                      verbose=0,
                                      index_col=None,
                                      sep=r',|;|\t')
        # reset the index
        buyer_to_seller['time'] = buyer_to_seller.index
        buyer_to_seller.reset_index(inplace=True, drop=True)
        # filter non-conformant rows
        buyer_to_seller_cols = ['source','target','rating','user_path','useless1','useless2','useless3','time']
        buyer_to_seller['row_len'] = buyer_to_seller.apply(lambda row: len(row), axis=1)
        buyer_to_seller = buyer_to_seller[buyer_to_seller['row_len'] == len(buyer_to_seller_cols)]
        # columns
        buyer_to_seller.columns = buyer_to_seller_cols + ['row_len']
        buyer_to_seller.drop(columns=['user_path','useless1','useless2','useless3','row_len'], inplace=True)
        
        # filter non-conformant rows
        seller_to_buyer_cols = ['time','source','target','rating','useless1','useless2','useless3','useless4','useless5']
        seller_to_buyer['row_len'] = seller_to_buyer.apply(lambda row: len(row), axis=1)
        seller_to_buyer = seller_to_buyer[seller_to_buyer['row_len'] == len(seller_to_buyer_cols)]
        # columns
        seller_to_buyer.columns = seller_to_buyer_cols + ['row_len']
        seller_to_buyer.drop(columns=['useless1','useless2','useless3','useless4','useless5','row_len'], inplace=True)
        # Filter rows based on the condition that "time" should be in a valid date format
        buyer_to_seller['time'] = pd.to_datetime(buyer_to_seller['time'], format='%m/%d/%y', errors='coerce')
        buyer_to_seller = buyer_to_seller[buyer_to_seller['time'].notnull()]
        
        seller_to_buyer['time'] = pd.to_datetime(seller_to_buyer['time'], format='%m/%d/%y', errors='coerce')
        seller_to_buyer = seller_to_buyer[seller_to_buyer['time'].notnull()]
        # extract year
        buyer_to_seller['year'] = buyer_to_seller.time.apply(lambda x : x.year)
        seller_to_buyer['year'] = seller_to_buyer.time.apply(lambda x : x.year)
        # concat the two datasets
        ratings = pd.concat([buyer_to_seller, seller_to_buyer])
        # retain only desired history
        ratings = ratings[(ratings.year >= self.begin_t) & (ratings.year <= self.end_t)]
        # change type of rating to float
        ratings['target'] = ratings['target'].replace('[^0-9.]', np.nan, regex=True)
        ratings['source'] = ratings['source'].replace('[^0-9.]', np.nan, regex=True)
        ratings['rating'] = ratings['rating'].replace('[^0-9.]', np.nan, regex=True)
        ratings.dropna(inplace=True)
        ratings['rating'] = ratings['rating'].astype(float)
        ratings['source'] = ratings['source'].astype(str).str.replace('.', '').astype(int)
        ratings['target'] = ratings['target'].astype(str).str.replace('.', '').astype(int)
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
            for node in nodes[:500]:#len(nodes) * .05)]:
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
        return 1 if np.sum(ratings) >= 10 else 0      

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
    
        
        