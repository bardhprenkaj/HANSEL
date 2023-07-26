import sys
from os import walk
import os
from src.evaluation.dynamic_graphs.dynamic_evaluator_manager import DynamicEvaluatorManager


path = sys.argv[1]
filenames = next(walk(path), (None, None, []))[2]

for i, file in enumerate(filenames):
    config_file_path = os.path.join(sys.argv[1], file)
    print('Executing:'+config_file_path)

    print('Creating the evaluation manager.......................................................')
    eval_manager = DynamicEvaluatorManager(config_file_path, K=10, run_number=i)
    print('Creating the evaluators...................................................................')
    eval_manager.create_evaluators()

    print('Evaluating the explainers..................................................................')
    eval_manager.evaluate()
