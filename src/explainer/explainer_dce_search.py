from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.explainer.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.oracle.oracle_base import Oracle
import numpy as np
import sys

class DCESearchExplainer(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance"""

    def __init__(self, id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self._gd = instance_distance_function
        self._name = 'DCESearchExplainer'


    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        l_input_inst = oracle.predict(instance)

        # if the method does not find a counterfactual example returns the original graph
        min_counterfactual = instance
        
        min_counterfactual_dist = sys.float_info.max

        for d_inst in dataset.instances:

            l_data_inst = oracle.predict(d_inst)

            if (l_input_inst != l_data_inst):
                d_inst_dist = self._gd.evaluate(instance, [d_inst], oracle)

                if (np.min(d_inst_dist) < min_counterfactual_dist):
                    min_counterfactual_dist = np.min(d_inst_dist)
                    min_counterfactual = d_inst
        
        return [min_counterfactual]


class DCESearchExplainerOracleless(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance. The oracleless version uses the labels in the labeled dataset instead 
    of querying the oracle"""

    def __init__(self, id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self._gd = instance_distance_function
        self._name = 'DCESearchExplainerOracleless'


    def explain(self, instance, oracle: Oracle, dataset: Dataset):
        l_input_inst = instance.graph_label

        # if the method does not find a counterfactual example returns the original graph
        min_counterfactual = instance

        min_counterfactual_dist = sys.float_info.max

        for d_inst in dataset.instances:
            
            l_data_inst = d_inst.graph_label

            if (l_input_inst != l_data_inst):
                d_inst_dist = self._gd.evaluate(instance, d_inst, oracle)

                if (d_inst_dist < min_counterfactual_dist):
                    min_counterfactual_dist = d_inst_dist
                    min_counterfactual = d_inst
        
        return min_counterfactual


