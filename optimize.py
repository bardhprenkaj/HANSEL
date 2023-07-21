import sys
from src.evaluation.dynamic_graphs.dynamic_evaluator_manager import DynamicEvaluatorManager

import wandb
import numpy as np 
import json

import traceback

config_file_path = sys.argv[1]
run_name = sys.argv[2]
project_name = sys.argv[3]

print('Executing:'+sys.argv[1])

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': f'Run={run_name}',
    'metric': {'goal': 'maximize', 'name': 'Correctness@1'},
    'parameters': 
    {
        'batch_size': {'values': [2,4,8,16,32,64]},
        'lr': {'values': [1e-4, 1e-3, 1e-2]},
        'epochs': {'values': list(range(5, 51, 5))},
        'k': {'values': list(range(1,21))},
        'out_channels': {'values': list(range(1,11))},
        'weight_decayer_alpha': {'values': ['no_decay', 'linear_decay', 'tolerance_scheduler']},
        'weight_decayer_beta': {'values': ['no_decay', 'linear_decay', 'tolerance_scheduler']},
        'init_weight_alpha': {'values': list(np.arange(0.1,1,0.1))},
        'alpha_step':{'values': list(np.arange(0.01,0.1,0.01))},
        'beta_step': {'values': list(np.arange(0.01,0.1,0.01))},
        'upper_bound_alpha': {'values': list(range(1,11))},
        'upper_bound_beta': {'values': list(range(1,11))},
        'lower_bound_alpha': {'values': list(range(1,11))},
        'lower_bound_beta': {'values': list(range(1,11))},
        'tolerance_alpha': {'values': [1e-4,1e-3,1e-2]},
        'tolerance_beta': {'values': [1e-4,1e-3,1e-2]}
    }
}



# Initialize sweep by passing in config.
sweep_id = wandb.sweep(
sweep=sweep_configuration, 
project=project_name
)


# sweep through the folds
def main():
    for fold_id in range(1):
        run = wandb.init()
        # note that we define values from `wandb.config`  
        # instead of defining hard values
        batch_size = wandb.config.batch_size
        lr = wandb.config.lr
        epochs = wandb.config.epochs
        k = wandb.config.k
        out_channels = wandb.config.out_channels
        weight_decay_alpha = wandb.config.weight_decayer_alpha
        weight_decay_beta = wandb.config.weight_decayer_beta
        init_weight_alpha = wandb.config.init_weight_alpha
        increment_step_alpha = wandb.config.alpha_step
        increment_step_beta = wandb.config.beta_step
        upper_bound_alpha = wandb.config.upper_bound_alpha
        upper_bound_beta = wandb.config.upper_bound_beta
        tolerance_alpha = wandb.config.tolerance_alpha
        tolerance_beta = wandb.config.tolerance_beta
        lower_bound_alpha = wandb.config.lower_bound_alpha
        lower_bound_beta = wandb.config.lower_bound_beta
        
        
        explainer_params = None
        with open(config_file_path, 'r') as f:
            explainer_params = json.load(f)['explainers'][0]['parameters']
        
        enc_name = explainer_params.get('encoder_name', 'gcn_encoder')
        dec_name = explainer_params.get('decoder_name', None)
        autoencoder_name = explainer_params.get('autoencoder_name', 'contrastive_gae')
        in_channels = explainer_params.get('in_channels', 4)
        num_classes = explainer_params.get('num_classes', 2)
        
        weight_dict = [
            {
                "name": f"{weight_decay_alpha}",
                "parameters" : {
                    "init_weight": init_weight_alpha,
                    "increment_step": increment_step_alpha,
                    "upper_bound": upper_bound_alpha,
                    "lower_bound": lower_bound_alpha,
                    "tolerance": tolerance_alpha
                }
            },
            {
                "name": f"{weight_decay_beta}",
                "parameters": {
                    "init_weight": 1-init_weight_alpha,
                    "increment_step": increment_step_beta,
                    "lower_bound": lower_bound_beta,
                    "upper_bound": upper_bound_beta,
                    "tolerance": tolerance_beta
                }
            }
        ]
        
        print('Creating the evaluation manager.......................................................')
        eval_manager = DynamicEvaluatorManager(config_file_path, run_number=0, K=k)
        eval_manager.create_evaluators()
        
        autoencoders = eval_manager._explainer_factory._autoencoder_factory.init_autoencoders(autoencoder_name=autoencoder_name,
                                                                                              enc_name=enc_name,
                                                                                              dec_name=dec_name,
                                                                                              in_channels=in_channels,
                                                                                              out_channels=out_channels,
                                                                                              num_classes=num_classes)
                
        schedulers = tuple([eval_manager._explainer_factory._weight_scheduler_factory.get_scheduler_by_name(d) for d in weight_dict])
        alpha_scheduler, beta_scheduler = schedulers 
    
        print('Creating the evaluators...................................................................')
        eval_manager.explainers[0].fold_id = fold_id
        eval_manager.explainers[0].lr = lr
        eval_manager.explainers[0].batch_size = batch_size
        eval_manager.explainers[0].epochs = epochs
        eval_manager.explainers[0].autoencoders = autoencoders
        eval_manager.explainers[0].k = k
        eval_manager.explainers[0].alpha_scheduler = alpha_scheduler
        eval_manager.explainers[0].beta_scheduler = beta_scheduler
        
        metric_reports = {f'{metric._name}@{i}': [] for metric in eval_manager.evaluation_metrics for i in range(1,k+1)}

        
        for i in range(len(eval_manager.evaluators)):
            eval_manager.evaluators[i].break_on_first = True
               
        print('Evaluating the explainers..................................................................')
        eval_manager.evaluate()
        
        
        for evaluator in eval_manager.evaluators:
            for metric in eval_manager.evaluation_metrics:
                for i in range(1, k+1):
                    for reported_vals in evaluator._results[f'{metric._name}@{i}']:
                        if isinstance(reported_vals, list):
                            metric_reports[f'{metric._name}@{i}'] += reported_vals
                        else:
                            metric_reports[f'{metric._name}@{i}'].append(reported_vals)    
    for i in range(1, k+1):
        values = metric_reports[f'{metric._name}@{i}']
        wandb.log({
            f'{metric._name}@{i}': sum(values)/len(values) for metric in eval_manager.evaluation_metrics
        })

# Start the sweep job
wandb.agent(sweep_id, function=main, count=10)