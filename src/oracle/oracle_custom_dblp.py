import os
from typing import Dict
from src.dataset.dataset_base import Dataset

import jsonpickle
import networkx as nx
import numpy as np

from src.dataset.data_instance_base import DataInstance
from src.oracle.oracle_base import Oracle


class DBLPCoAuthorshipCustomOracle(Oracle):

    def __init__(self, id, oracle_store_path, first_train_timestamp=2000, percentile=75, fold_id=0, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self._name = 'dblp_coauthorship_custom_oracle'
        self.fold_id = fold_id        
        self.percentile = percentile
        self.first_train_timestamp = first_train_timestamp

    def fit(self, dataset: Dataset, split_i=-1):
        print(dataset)
        self._name = f'{self._name}_fit_on_{dataset.name}_fold_id={self.fold_id}'
        # If there is an available oracle trained on that dataset load it
        if os.path.exists(os.path.join(self._oracle_store_path, self._name)):
            self.read_oracle(self._name)
        else:
            self.weight_dict: Dict[int: float] = {}
            for instance in dataset.instances:
                weights = list(nx.get_edge_attributes(instance.graph, 'weight').values())
                self.weight_dict[instance.id] = np.mean(weights) if len(weights) > 0 else 0
            self.percentile_value = np.percentile(list(self.weight_dict.values()), self.percentile)
            
            self.write_oracle()

    def _real_predict(self, data_instance: DataInstance):
        weights = list(nx.get_edge_attributes(data_instance.graph, 'weight').values())
        mean_weights = np.mean(weights) if len(weights) > 0 else 0
        return 1 if mean_weights > self.percentile_value else 0
        
    def _real_predict_proba(self, data_instance):
        return np.array([0, 1]) if self._real_predict(data_instance) else np.array([1, 0])

    def embedd(self, instance):
        return instance

    def write_oracle(self):
        directory = os.path.join(self._oracle_store_path, self._name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        with open(os.path.join(directory, 'weights.json'), 'w') as f:
            f.write(jsonpickle.encode(self.weight_dict))
            
        with open(os.path.join(directory, 'percentile_value.json'), 'w') as f:
            f.write(jsonpickle.encode({'percentile_value': self.percentile_value}))      

    def read_oracle(self, oracle_name):
        directory = os.path.join(self._oracle_store_path, oracle_name)
        
        weight_file_path = os.path.join(directory, 'weights.json')
        if os.path.exists(weight_file_path):
            with open(weight_file_path, 'r') as f:
                self.weight_dict = jsonpickle.decode(f.read())
                self.weight_dict = {int(k): v for k, v in self.weight_dict.items()}
                
        percentile_file_path = os.path.join(directory, 'percentile_value.json')
        if os.path.exists(percentile_file_path):
            with open(percentile_file_path, 'r') as f:
                self.percentile_value = float(jsonpickle.decode(f.read())['percentile_value'])
        