from typing import List

from src.dataset.data_instance_base import DataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.oracle.oracle_base import Oracle


class FidelityMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Fidelity'

    def evaluate(self, instance_1: DataInstance, other_instances: List[DataInstance], oracle: Oracle):

        label_instance_1 = oracle.predict(instance_1)
        oracle._call_counter -= 1
        
        labels = list()
        for instance in other_instances:
            label_instance_2 = oracle.predict(instance)
            oracle._call_counter -= 1
            labels.append(label_instance_2)
            
        prediction_fidelity = 1 if (label_instance_1 == instance_1.graph_label) else 0
        
        counterfactual_fidelity = 1 if (instance_1.graph_label in labels) else 0

        result = prediction_fidelity - counterfactual_fidelity
        
        return result