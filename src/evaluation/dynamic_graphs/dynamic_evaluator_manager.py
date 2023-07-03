

from src.dataset.dataset_factory import DatasetFactory
from src.dataset.dynamic_graphs.dataset_dynamic import DynamicDataset
from src.evaluation.dynamic_graphs.dynamic_evaluator import DynamicEvaluator
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.evaluation.evaluator_manager import EvaluatorManager
from src.explainer.explainer_factory import ExplainerFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory


class DynamicEvaluatorManager(EvaluatorManager):
    
    def __init__(self,
                 config_file_path,
                 K=5,
                 run_number=0,
                 dataset_factory: DatasetFactory = None,
                 embedder_factory: EmbedderFactory = None,
                 oracle_factory: OracleFactory = None,
                 explainer_factory: ExplainerFactory = None,
                 evaluation_metric_factory: EvaluationMetricFactory = None) -> None:
        
        super().__init__(config_file_path=config_file_path,
                         K=K,
                         run_number=run_number,
                         dataset_factory=dataset_factory,
                         embedder_factory=embedder_factory,
                         oracle_factory=oracle_factory,
                         explainer_factory=explainer_factory,
                         evaluation_metric_factory=evaluation_metric_factory)
        
        
    def create_evaluators(self):
        """Creates one evaluator for each combination of dataset-oracle-explainer using the chosen metrics
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        dataset_dicts = self._config_dict['datasets']
        oracle_dicts = self._config_dict['oracles']
        metric_dicts = self._config_dict['evaluation_metrics']
        explainer_dicts = self._config_dict['explainers']

        # Create the datasets
        for dataset_dict in dataset_dicts:
            self.datasets.append(self._dataset_factory.get_dataset_by_name(dataset_dict))

        # Create the evaluation metrics
        for metric_dict in metric_dicts:
            eval_metric = self._evaluation_metric_factory.get_evaluation_metric_by_name(metric_dict)
            self.evaluation_metrics.append(eval_metric)

        for explainer_dict in explainer_dicts:
            explainer = self._explainer_factory.get_explainer_by_name(explainer_dict, self._evaluation_metric_factory)
            self.explainers.append(explainer)

        evaluator_id = 0
        for dataset in self.datasets:
            
            assert isinstance(dataset, DynamicDataset)
            
            for explainer in self.explainers:
                for oracle_dict in oracle_dicts:

                    # The get_oracle_by_name method returns a fitted oracle
                    oracle = self._oracle_factory.get_oracle_by_name(oracle_dict,
                                                                     dataset,
                                                                     self._embedder_factory)

                    # Creating the evaluator
                    evaluator = DynamicEvaluator(evaluator_id, dataset,
                                                 oracle, explainer, self.evaluation_metrics,
                                                 self._output_store_path, self._run_number, self.K)

                    # Adding the evaluator to the evaluator's list
                    self.evaluators.append(evaluator)

                    # increasing the evaluator id counter
                    evaluator_id +=1
