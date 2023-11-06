import sys
from src.evaluation.dynamic_graphs.dynamic_evaluator_manager import DynamicEvaluatorManager

if __name__ == '__main__':
    i = int(sys.argv[2])
    config_file_path = sys.argv[1]
    
    print('Executing:'+config_file_path)

    print('Creating the evaluation manager.......................................................')
    eval_manager = DynamicEvaluatorManager(config_file_path, K=10, run_number=i)
    print('Creating the evaluators...................................................................')
    eval_manager.create_evaluators()

    print('Evaluating the explainers..................................................................')
    eval_manager.evaluate()
