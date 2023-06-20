import os
from abc import ABC, abstractmethod
from typing import Dict, List

import networkx as nx
import pandas as pd

from src.dataset.dataset_base import Dataset


class DynamicDataset(ABC):
    
    def __init__(self, id, begin_t, end_t, config_dict=None) -> None:    
        super().__init__()

        assert (begin_t < end_t)
        
        self.id = id
        self.name = 'dynamic_graph_dataset'

        self.begin_t = begin_t
        self.end_t = end_t
        
        self.dynamic_graph:Dict[Dataset] = { 
                                            key: Dataset(key % self.begin_t, config_dict=config_dict)\
                                                for key in range(self.begin_t, self.end_t + 1) 
                                            }
        
    @abstractmethod
    def build_temporal_graph(self):
        pass
    
    def read_datasets(self, dataset_path, graph_format='edge_list'):
        for key, dataset in self.dynamic_graph.items():
            self.dynamic_graph[key] = dataset.read_data(os.path.join(dataset_path, key), graph_format=graph_format)
    
    def write_datasets(self, dataset_path, graph_format='edge_list'):
        for key, dataset in self.dynamic_graph.items():
            dataset.write_data(os.path.join(dataset_path, key), graph_format=graph_format)
        
    @abstractmethod
    def preprocess_datasets(self):
        pass
    
    def slice(self, x, y) -> List[Dataset]:
        assert (x <= y)
        return [self.dynamic_graph[key] for key in self.dynamic_graph if x <= key <= y]

        
    
        
    
    