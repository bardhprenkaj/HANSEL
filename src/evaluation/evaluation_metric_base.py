from abc import ABC, abstractmethod
from typing import List

import networkx as nx

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class EvaluationMetric(ABC):

    def __init__(self, config_dict=None) -> None:
        super().__init__()
        self._name = 'abstract_metric'
        self._config_dict = config_dict

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abstractmethod
    def evaluate(self, instance_1 : DataInstance, other_instances : List[DataInstance], oracle : Oracle=None):
        pass
    