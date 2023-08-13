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
        pass

    def _real_predict(self, data_instance: DataInstance):
        ratings = np.array(list(nx.get_edge_attributes(data_instance.graph, 'weight').values()))
        return 1 if np.sum(ratings < 0) > np.sum(ratings >= 0) else 0  
        
    def _real_predict_proba(self, data_instance):
        return np.array([0, 1]) if self._real_predict(data_instance) else np.array([1, 0])

    def embedd(self, instance):
        return instance

    def write_oracle(self):
        pass
            
    def read_oracle(self, oracle_name):
        pass