from typing import List

from src.dataset.data_instance_base import DataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.oracle.oracle_base import Oracle


class OracleAccuracyMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Oracle_Accuracy'

    def evaluate(self, instance_1: DataInstance, other_instances: List[DataInstance], oracle: Oracle):

        predicted_label_instance_1 = oracle.predict(instance_1)
        oracle._call_counter -= 1
        real_label_instance_1 = instance_1.graph_label

        result = 1 if (predicted_label_instance_1 == real_label_instance_1) else 0
        
        return result