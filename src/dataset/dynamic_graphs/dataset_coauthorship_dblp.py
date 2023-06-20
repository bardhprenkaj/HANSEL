from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.dataset.dataset_base import Dataset
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset


import networkx as nx
import numpy as np
import os
import pandas as pd

from typing import Dict, List

class CoAuthorshipDBLP(DynamicDataset):
    
    def __init__(self,
                 id,
                 begin_time,
                 end_time,
                 min_connections=3,
                 percentile=75,
                 config_dict=None) -> None:
        
        super().__init__(id, begin_time, end_time, config_dict)
        self.name = 'coauthorship_dblp'
        self.min_connections = min_connections
        self.percentile = percentile        
        
    def read_csv_file(self, dataset_path):
        # read the number of vertices in for each simplex
        vertices = pd.read_csv(os.path.join(dataset_path, 'coauth-DBLP-nverts.txt'), sep=' ', header=None)
        vertices = vertices.values.squeeze().tolist()
        # read the simplices
        simplices = pd.read_csv(os.path.join(dataset_path, 'coauth-DBLP-simplices.txt'), sep=' ', names=['Values'], header=None)
        simplices = simplices.values.squeeze().tolist()
        # group simplices according to the vertex number in vertices
        graphs = self.__group_simplices(vertices, simplices)
        graphs, indices_remaining = self.__filter_small_simplices(graphs)
        # read the timestamps
        times = pd.read_csv(os.path.join(dataset_path, 'coauth-DBLP-times.txt'), sep=' ', header=None, names=['timestamp'])
        # take only those that have a large enough simplex
        times = times.iloc[indices_remaining]
        
        times['graph'] = np.array(graphs)
        times = times.sort_values(by='timestamp')
        # retain only desired history
        times = times[(times.timestamp >= self.begin_t) & (times.timestamp <= self.end_t)]
        
        self.grouped_by_time = times.groupby('timestamp')
        
    
    def __group_simplices(self, A, B):
        C = []
        index = 0

        for num in A:
            group = B[index:index+num]
            C.append(group)
            index += num

        return C

    def __filter_small_simplices(self, C):
        indices = []
        result = []

        for i, sublist in enumerate(C):
            if len(sublist) >= self.min_connections:
                result.append(np.array(sublist))
                indices.append(i)

        return np.array(result), indices
    
    
    def build_temporal_graph(self):
        self.unprocessed_data = {}
        for row in self.grouped_by_time:
            G = nx.Graph()
            for graph in row[1]['graph'].to_numpy():
                G.add_nodes_from(graph)
                for i in range(len(graph)):
                    for j in range(i+1, len(graph)):
                        if G.has_edge(graph[i], graph[j]):
                            G[graph[i]][graph[j]]['weight'] += 1
                        else:
                            G.add_edge(graph[i], graph[j], weight=1)
                # year -> entire graph of coauthor simplices
                self.unprocessed_data[row[0]] = G
        
        self.preprocess_datasets()
                
                
    def preprocess_datasets(self):
        aligned_graphs_in_time = self.__trace_ego_networks(self.unprocessed_data)
        begin = min(aligned_graphs_in_time.keys())
        end = max(aligned_graphs_in_time.keys())
        
        for i in range(begin, end + 1):
            self.dynamic_graph[i % begin]._name = f'DBLP@{i}'
            labels = self.__get_labels(aligned_graphs_in_time, i)
            for node in aligned_graphs_in_time[begin].nodes():
                ego_net = nx.ego_graph(aligned_graphs_in_time[i], node)
                
                instance = DataInstanceWFeaturesAndWeights(node)
                instance.weights(nx.to_numpy_array(ego_net))
                instance.graph = nx.to_numpy_array(ego_net, weight=None)
                instance.graph_label = labels[node]
                
                self.dynamic_graph[i % begin].instances.append(instance)
                
        # clear those snapshots that are empty
        for i in range(self.begin_t, self.end_t + 1):
            if self.dynamic_graph[i % self.begin_t].get_data_len() == 0:
                self.dynamic_graph.pop(i % self.begin_t, None)
                
                
    def __trace_ego_networks(self, temporal_graph):
        begin = min(temporal_graph.keys())
        end = max(temporal_graph.keys())
        
        for node in temporal_graph[begin].nodes():
            for i in range(begin + 1, end + 1):
                if node not in temporal_graph[i].nodes():
                    temporal_graph[i] = nx.compose(temporal_graph[i],
                                                   nx.ego_graph(temporal_graph[i-1], node))
                    
        return temporal_graph
    
    
    def __in_percentile(self, average_weights: Dict[int, float]) -> List[int]:
        percentile_value = np.percentile(list(average_weights.values()), self.percentile)
        return [1 if num >= percentile_value else 0 for num in average_weights]
    
    
    def __get_labels(self, temporal_graph, year) -> Dict[int, int]:
        assert (year <= max(temporal_graph.keys()))
        begin = min(temporal_graph.keys())

        weight_dict = {}
        for node in temporal_graph[begin].nodes():
            ego = nx.ego_graph(temporal_graph[year], node)
            weights = list(nx.get_edge_attributes(ego, 'weight').values())
            weight_dict[node] = np.mean(weights)
            
        return dict(zip(temporal_graph[begin].nodes(),
                        self.__in_percentile(weight_dict)))

    
        
        