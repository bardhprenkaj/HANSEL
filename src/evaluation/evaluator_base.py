import os
import time
from abc import ABC
from typing import List

import json
import numpy as np

from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.explainer.explainer_base import Explainer
from src.oracle.oracle_base import Oracle


class Evaluator(ABC):

    def __init__(self, id, data: Dataset, oracle: Oracle, explainer: Explainer, evaluation_metrics, results_store_path, K=5, run_number=0) -> None:
        super().__init__()
        self._id = id
        self._name = 'Evaluator_for_' + explainer.name + '_using_' + oracle.name
        self._data = data
        self._oracle = oracle
        self._oracle.reset_call_count()
        self._explainer = explainer
        self._results_store_path = results_store_path
        self._evaluation_metrics = evaluation_metrics
        self._run_number = run_number
        self._explanations = []
        self._K = K
        # Building the config file to write into disk
        evaluator_config = {'dataset': data._config_dict, 'oracle': oracle._config_dict, 'explainer': explainer._config_dict, 'metrics': []}
        for metric in evaluation_metrics:
            evaluator_config['metrics'].append(metric._config_dict)
        # creatig the results dictionary with the basic info
        self._results = {'config':evaluator_config, 'runtime': []}

    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, new_K):
        self._K = new_K
    
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def dataset(self):
        return self._data

    @dataset.setter
    def dataset(self, new_dataset: Dataset):
        self._data = new_dataset

    @property
    def explanations(self):
        return self._explanations

    @explanations.setter
    def explanations(self, new_explanations_list):
        self._explanations = new_explanations_list
        
    @property
    def explainer(self):
        return self._explainer


    def get_instance_explanation_pairs(self):
        # Check if the explanations were generated already
        if len(self.explanations) < 1:
            return None

        # iterates over the original instances and the explanations
        n_ins = len(self.dataset.instances)
        result = []
        for i in range(0, n_ins):
            result.append((self.dataset.instances[i], self.explanations[i]))

        return result


    def get_instance_and_counterfactual_classifications(self):
        # Check if the explanations were generated already
        if len(self.explanations) < 1:
            return None

        # iterates over the original instances and the explanations
        n_ins = len(self.dataset.instances)
        result = []
        for i in range(0, n_ins):
            label_inst = self._oracle.predict(self.dataset.instances[i])
            label_cf = self._oracle.predict(self.explanations[i])
            self._oracle._call_counter -= 2 

            result.append({'instance_id': self.dataset.instances[i].id,
                             'ground_truth_label': self.dataset.instances[i].graph_label,
                             'instance_label': label_inst,
                             'counterfactual_label': label_cf})

        return result


    def evaluate(self):
        for m in self._evaluation_metrics:
            self._results[m._name] = []

        # If the explainer was trained then evaluate only on the test set, else evaluate on the entire dataset
        fold_id = self._explainer.fold_id
        if fold_id == -1:
            for inst in self._data.instances:
                                
                start_time = time.time()
                counterfactuals: List[DataInstance] = self._explainer.explain(inst, self._oracle, self._data)
                end_time = time.time()
                # giving the same id to the counterfactual and the original instance
                """for i in range(len(counterfactuals)):
                    counterfactuals[i].id = inst.id"""

                self._explanations += counterfactuals

                # The runtime metric is built-in inside the evaluator``
                self._results.setdefault('runtime', []).append(end_time - start_time)

                self._real_evaluate(inst, counterfactuals)
        else:
            self._results = {}
            test_indices = self.dataset.splits[fold_id]['test']
            test_set = [i for i in self.dataset.instances if i.id in test_indices]
            print(f'The length of the test set is = {len(test_set)}')

            for inst in test_set:
                                
                start_time = time.time()
                counterfactuals: List[DataInstance] = self._explainer.explain(inst, self._oracle, self._data)
                end_time = time.time()
                # giving the same id to the counterfactual and the original instance 
                """for i in range(len(counterfactuals)):
                    counterfactuals[i].id = inst.id"""
                    
                self._explanations.append(counterfactuals)

                # The runtime metric is built-in inside the evaluator``
                self._results.setdefault('runtime', []).append(end_time - start_time)

                self._real_evaluate(inst, counterfactuals)
                print('evaluated instance with id ', str(inst.id))

        self.write_results()


    def _real_evaluate(self, instance, counterfactuals: List[DataInstance], oracle = None):
        is_alt = False
        if (oracle is None):
            is_alt = True
            oracle = self._oracle
            
        for metric in self._evaluation_metrics:
            for k in range(1, self._K + 1):
                self._results.setdefault(f'{metric._name}@{k}', []).append(metric.evaluate(instance, counterfactuals[:k], oracle))

    def write_results(self):

        output_oracle_dataset_path = os.path.join(self._results_store_path, self._oracle.name)
        if not os.path.exists(output_oracle_dataset_path):
            os.mkdir(output_oracle_dataset_path)

        output_explainer_path = os.path.join(output_oracle_dataset_path, self._explainer.name)
        if not os.path.exists(output_explainer_path):
            os.mkdir(output_explainer_path)

        results_uri = os.path.join(output_explainer_path, 'results_run-' + str(self._run_number) + '.json')
        self._run_number += 1

        with open(results_uri, 'w') as results_writer:
            json.dump(self._results, results_writer)
            #results_writer.write(jsonpickle.encode(self._results))

