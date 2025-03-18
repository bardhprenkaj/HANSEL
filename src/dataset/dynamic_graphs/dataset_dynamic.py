import os
from abc import ABC, abstractmethod
from typing import Dict, List

from src.dataset.dataset_base import Dataset


class DynamicDataset(ABC):
    
    def __init__(self, id, begin_t, end_t, config_dict=None) -> None:    
        super().__init__()

        assert (begin_t <= end_t)
        
        self._id = id
        self._name = 'dynamic_graph_dataset'

        self.begin_t = begin_t
        self.end_t = end_t
        
        self.dynamic_graph:Dict[int, Dataset] = { 
                                            key: Dataset(key, config_dict=config_dict)\
                                                for key in range(self.begin_t, self.end_t + 1) 
                                            }
        
        self.splits = {
            key: None for key in range(self.begin_t, self.end_t + 1)
        }
        
        self._instance_id_counter = len(self.dynamic_graph)
        self._config_dict = config_dict
        
    @abstractmethod
    def build_temporal_graph(self):
        pass
    
    def read_datasets(self, dataset_path, graph_format='edge_list'):
        eliminate = []
        for key, dataset in self.dynamic_graph.items():
            print(f'Reading year {key}')
            dataset.read_data(os.path.join(dataset_path, f'{key}'), graph_format=graph_format)
            self.dynamic_graph[key] = dataset
            
    
    def write_datasets(self, dataset_path, graph_format='edge_list'):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
                    
        for dataset in self.dynamic_graph.values():
            dataset.write_data(dataset_path, graph_format=graph_format)
            
            
    def load_or_generate_splits(self, dataset_folder, n_splits=10, shuffle=True):
        for key, dataset in self.dynamic_graph.items():
            dataset_path = os.path.join(dataset_folder, 'processed', dataset._name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            splits_uri = os.path.join(dataset_path)
            n_splits = min(len(dataset.instances), n_splits)
            self.splits[key] = dataset.load_or_generate_splits(splits_uri,
                                                               n_splits=n_splits,
                                                               shuffle=shuffle)
        
    @abstractmethod
    def preprocess_datasets(self):
        pass
    
    def slice(self, x, y) -> List[Dataset]:
        assert (x <= y)
        return [self.dynamic_graph[key] for key in self.dynamic_graph if x <= key <= y]

        
    
        
    
    