from functools import lru_cache
from typing import List

from src.dataset.data_instance_base import DataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.oracle.oracle_base import Oracle


class SmilesLevenshteinMetric(EvaluationMetric):
    """Provides the ratio between the number of features modified to obtain the counterfactual example
     and the number of features in the original instance. Only considers structural features.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Smiles-Levenshtein'

    def evaluate(self, instance_1: DataInstance, other_instances: List[DataInstance], oracle: Oracle = None):
        return [self.lev_dist(instance_1.smiles, instance.smiles) for instance in other_instances]

    def lev_dist(self, a, b):
        '''
        This function will calculate the levenshtein distance between two input
        strings a and b
        
        params:
            a (String) : The first string you want to compare
            b (String) : The second string you want to compare
            
        returns:
            This function will return the distnace between string a and b.
            
        example:
            a = 'stamp'
            b = 'stomp'
            lev_dist(a,b)
            >> 1.0
        '''
        
        @lru_cache(None)  # for memorization
        def min_dist(s1, s2):

            if s1 == len(a) or s2 == len(b):
                return len(a) - s1 + len(b) - s2

            # no change required
            if a[s1] == b[s2]:
                return min_dist(s1 + 1, s2 + 1)

            return 1 + min(
                min_dist(s1, s2 + 1),      # insert character
                min_dist(s1 + 1, s2),      # delete character
                min_dist(s1 + 1, s2 + 1),  # replace character
            )

        return min_dist(0, 0)