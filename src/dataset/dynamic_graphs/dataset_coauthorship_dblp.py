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
                 min_nodes_per_egonet=5,
                 features_dim=8,
                 seed=42,
                 config_dict=None) -> None:
        
        super().__init__(id, begin_time, end_time, config_dict)
        self.name = 'coauthorship_dblp'
        self.percentile = percentile
        self.sampling_ratio = sampling_ratio
        self.min_nodes_per_egonet = min_nodes_per_egonet
        self.features_dim = features_dim
            
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
        
        times['graph'] = np.array(graphs, dtype=object)
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
        #self.__sample_on_first_graph()
        print("Tracing ego networks...")
        self.__trace_ego_networks(self.unprocessed_data)
        print('Eliminating the empty snapshots')
        # clear those snapshots that are empty
        for i in range(self.begin_t, self.end_t + 1):
            if self.dynamic_graph[i].get_data_len() == 0:
                self.dynamic_graph.pop(i, None)
        print(f'Finished preprocessing.')
                
                
    def __trace_ego_networks(self, temporal_graph):
        begin = min(temporal_graph.keys())
        end = max(temporal_graph.keys())
        
        unique_ego_nets = self.__sample_nodes(temporal_graph[begin],
                                              list(temporal_graph[begin].nodes()),
                                              sample=True)
        unique_ego_nets = self.__random_select_kv_pairs(unique_ego_nets)
        
        combined_graph = nx.Graph()
        for ego_network in unique_ego_nets.values():
            combined_graph.add_nodes_from(ego_network.nodes())
            combined_graph.add_edges_from(ego_network.edges())
            
        # create the data instances for the first timestamp
        temporal_graph[begin] = combined_graph
        labels = self.__get_labels(unique_ego_nets)
        self.dynamic_graph[begin]._name = f'{begin}'
        for node, graph in unique_ego_nets.items():
            self.__create_data_instance(id=node, graph=graph,
                                        year=begin, label=labels[node])
        
        # trace the ego networks through time
        for node, graph in unique_ego_nets.items():
            print(f'Working for node = {node} on ego network tracing...')
            for i in range(begin + 1, end + 1):
                if node not in temporal_graph[i].nodes():
                    temporal_graph[i] = nx.compose(temporal_graph[i], graph)
                    
            print(f'Finished tracing for node = {node}')
            
        for t in range(begin + 1, end + 1):
            # set the name of thedataset
            self.dynamic_graph[t]._name = f'{t}'
            # get the unique ego networks at time t
            unique_ego_nets_at_t = self.__sample_nodes(temporal_graph[t],
                                                       list(unique_ego_nets.keys()),
                                                       unique=False)
            # get the labels of the ego networks at time t
            labels = self.__get_labels(unique_ego_nets_at_t)
            # create data instances at time t
            for node, graph in unique_ego_nets_at_t.items():
                self.__create_data_instance(id=node, graph=graph,
                                            year=t, label=labels[node])
                      
    
    def __create_data_instance(self, id: int, graph: nx.Graph, year: int, label: float):
        instance = DataInstanceWFeaturesAndWeights(id=id)
        instance.name = f'ego_network_for_node={id}'
        instance.weights = nx.to_numpy_array(graph)
        instance.features = np.random.rand(graph.number_of_nodes(), self.features_dim)
        instance.graph = graph
        instance.graph_label = label
        print(f'Adding DataInstance with id = {id}')
        self.dynamic_graph[year].instances.append(instance)
    
    def __sample_nodes(self, graph: nx.Graph, trace_nodes: List[int], sample=False, unique=True):
        ego_networks = {}
        for node in trace_nodes:
            ego_network = nx.ego_graph(graph, node)
            if not sample or ego_network.number_of_nodes() >= self.min_nodes_per_egonet:
                ego_networks[node] = ego_network
        
        unique_networks = ego_networks
        
        if unique:
            unique_networks = {}
            for node, ego_network in ego_networks.items():
                is_duplicate = False
            
                for unique_network in unique_networks.values():
                    if set(ego_network.nodes()).issubset(set(unique_network.nodes())):
                        is_duplicate = True
                        break
            
                if not is_duplicate:
                    unique_networks[node] = ego_network
    
        return unique_networks
    
    def __random_select_kv_pairs(self, dictionary: Dict[int, nx.Graph]):
        keys = list(dictionary.keys())
        num_nodes_to_keep = int(self.sampling_ratio * len(keys))
        selected_keys = random.sample(keys, num_nodes_to_keep)
        selected_pairs = {key: dictionary[key] for key in selected_keys}
        return selected_pairs
        
    def __in_percentile(self, average_weights: Dict[int, float]) -> Dict[int, int]:
        percentile_value = np.percentile(list(average_weights.values()), self.percentile)
        return {key: 1 if value > percentile_value else 0 for key, value in average_weights.items()}
    
    def __get_labels(self, ego_nets: Dict[int, nx.Graph]) -> Dict[int, int]:      
        weight_dict = {}
        for node, graph in ego_nets.items():
            weights = list(nx.get_edge_attributes(graph, 'weight').values())
            weight_dict[node] = np.mean(weights) if len(weights) > 0 else -1
            
        return  self.__in_percentile(weight_dict)

    
        
        