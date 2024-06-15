from typing import List

from src.dataset.data_instance_base import DataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.oracle.oracle_base import Oracle
import json

from os.path import join

class CorrectnessMetric(EvaluationMetric):
    """Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def __init__(self, dump=False, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Correctness'
        self.dump = dump
        self._ged = GraphEditDistanceMetric()

    def evaluate(self, instance_1: DataInstance, other_instances: List[DataInstance], oracle: Oracle):
        oracle_results = list()
                
        label_instance_1 = oracle.predict(instance_1)
        oracle._call_counter -= 1
        for i, instance in enumerate(other_instances):
            label_instance_2 = oracle.predict(instance)
            oracle._call_counter -= 1
            oracle_results.append(True if label_instance_1 != label_instance_2 else False)
            if self.dump:
                if i == 0 and label_instance_1 != label_instance_2:
                    with open(join(instance_1.dataset, f'cf_for_id={instance_1.id}_is_id={instance.id}.json'), 'w') as f:
                        d = {'original': instance_1.to_numpy_array().tolist(),
                            'cf': instance.to_numpy_array().tolist()}
                        json.dump(d, f)
                elif i == 0 and label_instance_1 == label_instance_2:
                    with open(join(instance_1.dataset, f'cf_for_id={instance_1.id}_is_id={instance.id}.json'), 'w') as f:
                        d = {'original': instance_1.to_numpy_array().tolist(),
                            'cf': instance_1.to_numpy_array().tolist()}
                        json.dump(d, f)
                    
        geds = self._ged.evaluate(instance_1, other_instances, oracle)

        return 1 if any(oracle_results) and any(geds) else 0