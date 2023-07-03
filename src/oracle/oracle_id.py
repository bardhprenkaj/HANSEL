from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle

import numpy as np

class IDOracle(Oracle):

    def __init__(self, id, oracle_store_path, config_dict=None) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        

    def fit(self, dataset: Dataset, split_i=-1):
       pass
   
    def _real_predict(self, data_instance):
        return data_instance.graph_label
    
    def _real_predict_proba(self, data_instance):
        return np.array([1,0]) if self._real_predict(data_instance) else np.array([0,1])

    def embedd(self, instance):
        return instance

    def write_oracle(self):
        pass

    def read_oracle(self, oracle_name):
        pass
