from src.dataset.dataset_base import Dataset
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle
from src.evaluation.evaluator_base import Evaluator
from copy import deepcopy
import numpy as np

class DynamicEvaluator(Evaluator):
    
    def __init__(self,
                 id, 
                 data: Dataset,
                 oracle: Oracle,
                 explainer: Explainer,
                 evaluation_metrics,
                 results_store_path,
                 run_number=0,
                 break_on_first=False,
                 K=5) -> None:
        
        assert isinstance(data, DynamicDataset)
        
        super().__init__(id, data, oracle,
                         explainer, evaluation_metrics,
                         results_store_path, K, run_number)
        
        self.dyn_graph = data.dynamic_graph
        self.break_on_first = break_on_first
        
        
    def evaluate(self):
        begin_time = min(self.dyn_graph.keys())
        
        for time in self.dyn_graph.keys():
            print(f'Evaluating for iteration {time - begin_time}')
            self._explainer.iteration = time - begin_time
            self._data = self.dyn_graph[time]
            super().evaluate()
            if self._explainer.iteration > 0:
                self._explainer.update()
            if self.break_on_first:
                break