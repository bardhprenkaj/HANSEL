from src.dataset.dataset_base import Dataset
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle
from src.evaluation.evaluator_base import Evaluator
from copy import deepcopy

class DynamicEvaluator(Evaluator):
    
    def __init__(self,
                 id, 
                 data: Dataset,
                 oracle: Oracle,
                 explainer: Explainer,
                 evaluation_metrics,
                 results_store_path,
                 run_number=0) -> None:
        
        assert isinstance(data, DynamicDataset)
        
        super().__init__(id, data, oracle,
                         explainer, evaluation_metrics,
                         results_store_path, run_number)
        
        self.dataset = deepcopy(data)
        
        
    def evaluate(self):
        begin_time = min(self.dataset.dynamic_graph.keys())
        for time in self.dataset.dynamic_graph.keys():
            self._explainer.time = time - begin_time
            self._data = self.dataset.dynamic_graph[time]
            super().evaluate()                
                
                