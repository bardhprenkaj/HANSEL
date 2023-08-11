import os
from typing import Dict
from src.dataset.dataset_base import Dataset

import jsonpickle
import networkx as nx
import numpy as np

from src.dataset.data_instance_base import DataInstance
from src.oracle.oracle_base import Oracle


class BTCAlphaCustomOracle(Oracle):

    def __init__(self, id, oracle_store_path, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self._name = 'btc_alpha_custom_oracle'

    def fit(self, dataset: Dataset, split_i=-1):
        self._name = f'{self._name}_fit_on_{dataset.name}'
        # If there is an available oracle trained on that dataset load it
        if os.path.exists(os.path.join(self._oracle_store_path, self._name)):
            self.read_oracle(self._name)
        else:
            ratings = []
            for instance in dataset.instances:
                ratings += list(nx.get_edge_attributes(instance.graph, 'weight').values())
            self.mean_weights = np.mean(ratings)
            
            self.write_oracle()

    def _real_predict(self, data_instance: DataInstance):
        ratings = np.array(list(nx.get_edge_attributes(data_instance.graph, 'weight').values()))
        return 1 if np.sum(ratings < 0) else 0
        
    def _real_predict_proba(self, data_instance):
        return np.array([0, 1]) if self._real_predict(data_instance) else np.array([1, 0])

    def embedd(self, instance):
        return instance

    def write_oracle(self):
        directory = os.path.join(self._oracle_store_path, self._name)
        if not os.path.exists(directory):
            os.mkdir(directory)
                        
        with open(os.path.join(directory, 'mean_weights.json'), 'w') as f:
            f.write(jsonpickle.encode({'mean_weights': self.mean_weights}))
            
    def read_oracle(self, oracle_name):
        directory = os.path.join(self._oracle_store_path, oracle_name)
                
        weight_file_path = os.path.join(directory, 'mean_weights.json')
        if os.path.exists(weight_file_path):
            with open(weight_file_path, 'r') as f:
                self.mean_weights = float(jsonpickle.decode(f.read())['mean_weights'])