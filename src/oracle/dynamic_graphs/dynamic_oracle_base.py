from src.oracle.embedder_base import Embedder
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset
from src.oracle.oracle_base import Oracle


class DynamicOracle(Oracle):
    
    def __init__(self,
                 id,
                 base_oracle: Oracle,
                 oracle_store_path,
                 config_dict=None) -> None:
        
        super().__init__(id=id, oracle_store_path=oracle_store_path, config_dict=config_dict)
        self._name = 'dynamic_oracle'
        self.base_oracle = base_oracle
        
        
    def fit(self, dataset: DynamicDataset, timestamp=-1):
        snapshot_dataset = dataset.dynamic_graph.get(timestamp, None)
        if snapshot_dataset:
            # fit the base model with the current snapshot
            self.base_oracle.fit(snapshot_dataset, -1)
                        
            
    def _real_predict(self, data_instance):
        return self.base_oracle.predict(data_instance)
    
    def _real_predict_proba(self, data_instance):
        return self.base_oracle.predict_proba(data_instance)
    
    def embedd(self, instance):
        return instance
    
    def write_oracle(self):
        return self.base_oracle.write_oracle()
    
    def read_oracle(self, oracle_name):
        return self.base_oracle.read_oracle(self.base_oracle_model._name)
