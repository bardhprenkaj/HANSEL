from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.dataset.dataset_base import Dataset
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset


import networkx as nx
import numpy as np
import os
import pandas as pd
import random
from typing import Dict, List

class CoAuthorshipDBLP(DynamicDataset):
    
    def __init__(self,
                 id,
                 begin_time,
                 end_time,
                 percentile=75,
                 sampling_ratio=.25,
                 seed=42,
                 config_dict=None) -> None:
        
        super().__init__(id, begin_time, end_time, config_dict)
        self.name = 'coauthorship_dblp'
        self.percentile = percentile
        self.sampling_ratio = sampling_ratio
            
        random.seed(seed)       
        
    def read_csv_file(self, dataset_path):
        # read the number of vertices in for each simplex
        vertices = pd.read_csv(os.path.join(dataset_path, 'coauth-DBLP-nverts.txt'), sep=' ', header=None)
        vertices = vertices.values.squeeze().tolist()
        # read the simplices
        simplices = pd.read_csv(os.path.join(dataset_path, 'coauth-DBLP-simplices.txt'), sep=' ', names=['Values'], header=None)
        simplices = simplices.values.squeeze().tolist()
        # group simplices according to the vertex number in vertices
        graphs = self.__group_simplices(vertices, simplices)
        # read the timestamps
        times = pd.read_csv(os.path.join(dataset_path, 'coauth-DBLP-times.txt'), sep=' ', header=None, names=['timestamp'])
        # take only those that have a large enough simplex
        
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

        return np.array(result, dtype=object), indices
    
    
    def build_temporal_graph(self):
        self.unprocessed_data = {}
        for row in self.grouped_by_time:
            print(f'Working for time={row[0]}')
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
        print("Preprocessing began...")
        self.__sample_on_first_graph()
        print("Tracing ego networks...")
        aligned_graphs_in_time = self.__trace_ego_networks(self.unprocessed_data)
        begin = min(aligned_graphs_in_time.keys())
        end = max(aligned_graphs_in_time.keys())
        print('Finished tracing ego networks')
        
        print('Converting all networkx to DataInstance objects...')
        for i in range(begin, end + 1):
            print(f'Working for year={i}')
            self.dynamic_graph[i]._name = f'DBLP@{i}'
            labels = self.__get_labels(aligned_graphs_in_time, i)
            print(labels)
            for node in aligned_graphs_in_time[begin].nodes():
                ego_net = nx.ego_graph(aligned_graphs_in_time[i], node)
                                
                instance = DataInstanceWFeaturesAndWeights(id=node)
                instance.name = f'ego_network_for_node={node}'
                instance.weights = nx.to_numpy_array(ego_net)
                instance.graph = ego_net
                instance.graph_label = labels[node]
                
                print(f'Adding DataInstance with id = {node}')
                self.dynamic_graph[i].instances.append(instance)
                
        print('Eliminating the empty snapshots')
        # clear those snapshots that are empty
        for i in range(self.begin_t, self.end_t + 1):
            if self.dynamic_graph[i].get_data_len() == 0:
                self.dynamic_graph.pop(i % self.begin_t, None)
                
        print(f'Finished preprocessing.')
                
                
    def __trace_ego_networks(self, temporal_graph):
        begin = min(temporal_graph.keys())
        end = max(temporal_graph.keys())
        
        for node in temporal_graph[begin].nodes():
            print(f'Working for node = {node} on ego network tracing...')
            for i in range(begin + 1, end + 1):
                if node not in temporal_graph[i].nodes():
                    temporal_graph[i] = nx.compose(temporal_graph[i],
                                                   nx.ego_graph(temporal_graph[i-1], node))
            print(f'Finished tracing for node = {node}')
        return temporal_graph
    
    
    def __sample_on_first_graph(self):
        begin = min(self.unprocessed_data.keys())
        num_nodes_to_keep = int(self.sampling_ratio * self.unprocessed_data[begin].number_of_nodes())
        
        print(f'Number of nodes to keep = {num_nodes_to_keep}')
        nodes_to_keep = random.sample(self.unprocessed_data[begin].nodes(), num_nodes_to_keep)
        subgraph = self.unprocessed_data[begin].subgraph(nodes_to_keep)
        self.unprocessed_data[begin] = subgraph

    
    def __in_percentile(self, average_weights: Dict[int, float]) -> Dict[int, int]:
        percentile_value = np.percentile(list(average_weights.values()), self.percentile)
        return {key: 1 if value > percentile_value else 0 for key, value in average_weights.items()}
    
    def __get_labels(self, temporal_graph, year) -> Dict[int, int]:
        assert (year <= max(temporal_graph.keys()))
        begin = min(temporal_graph.keys())

        weight_dict = {}
        for node in temporal_graph[begin].nodes():
            ego = nx.ego_graph(temporal_graph[year], node)
            weights = list(nx.get_edge_attributes(ego, 'weight').values())
            weight_dict[node] = np.mean(weights) if len(weights) > 0 else 0
            
        return  self.__in_percentile(weight_dict)

    
        
        