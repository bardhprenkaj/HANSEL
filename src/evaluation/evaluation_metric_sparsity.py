from typing import List

import numpy as np

from src.dataset.data_instance_base import DataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.oracle.oracle_base import Oracle


class SparsityMetric(EvaluationMetric):
    """Provides the ratio between the number of features modified to obtain the counterfactual example
     and the number of features in the original instance. Only considers structural features.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Sparsity'

    def evaluate(self, instance_1: DataInstance, other_instances: List[DataInstance], oracle: Oracle = None):
        ged = GraphEditDistanceMetric()
        result = np.array(ged.evaluate(instance_1, other_instances, oracle))/self.number_of_structural_features(instance_1)
        return list(result)

    def number_of_structural_features(self, data_instance : DataInstance) -> float:
        return len(data_instance.graph.edges) + len(data_instance.graph.nodes)

