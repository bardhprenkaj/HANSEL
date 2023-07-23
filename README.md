
# HANSEL: Graph Counterfactual Explanation under Dynamic Data Changes

## Table of Contents

* [Team Information](#team-information)
* [General Information](#general-information)
* [Citation Request](#citation-request)
* [Requirements](#requirements)
* [Install](#installation)
* [Resources Provided with the Framework](#resources-provided-with-the-framework)
* [How to Use](#how-to-use)
* [References](#references)

## Team Information:
* Bardh Prenkaj (Sapienza University of Rome) [Principal Investigator]
* Mario Villaizán-Vallelado (University of Valladolid) [Investigator]
* Tobias Leemann (University of Tuebingen) [Investigator]
* Gjergji Kasneci (TU Munich) [Supervisor]

## General Information:

HANSEL is an open source framework for Evaluating Graph Counterfactual Explanation Methods under data changes (i.e. data distribution drifts). Our main goal is to create a generic platform that allows the researchers to speed up the process of developing and testing new Timed-Graph Counterfactual Explanation Methods.


## Citation Request:

Please cite our paper if you use HANSEL in your experiments:

Prenkaj, B.; Villaiz ́an-Vallelado, M.; Leemann, T.; and Kasneci, G. 2023. Adapting to Change: Robust Counterfactual Explanations in Dynamic Data Landscapes. In Proceedings of The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery (ECML PKDD 2023)

```latex:
@inproceedings{prenkaj2023dygrace,
  title={Adapting to Change: Robust Counterfactual Explanations in Dynamic Data Landscapes},
  author={Bardh Prenkaj and Mario Villaizán-Vallelado and Tobias Leemann and Gjergji Kasneci},
  booktitle={Proceedings of The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery (ECML/PKDD 2023)},
  year={2023}
}
```

## Requirements:

* scikit-learn
* numpy 
* scipy
* pandas
* tensorflow (for GCN)
* jsonpickle (for serialization)
* torch
* joblib
* rdkit (Molecules)
* exmol (maccs method)
* networkx (Graphs)

## Installation:
The easiest way to get HANSEL up and running with all the dependencies is to pull the development Docker image available in [Docker Hub](https://hub.docker.com/):

```
docker pull gretel/gretel:latest
```

The image is based on `tensorflow/tensorflow:latest-gpu` and it's GPU ready. In order to setup the container we recommend you to run:

```
docker-compose run gretel
```

For simplicity we provide several **makefile** rules for easy interaction with the Docker interface:

 * `make docker` - builds the development image from scratch
 * `make pull` - pull the development image
 * `make push` - push the development image
 * `make demo` - run the demo in the development image.

## Resources provided with the Framework:

### Datasets:

* **DynTree-Cycles**: Synthetic data set where each instance is a graph. The instance can be either a tree or a tree with several cycle patterns connected to the main graph by one edge

* **DBLP-Coauthors**: It contains graphs where the vertices represent authors, and the connections are co-authorship relationships between two authors.

### Oracles:

* **KNN**

* **SVM**

* **GCN**

* **CustomOracle**

### Explainers:

* **DyGRACE**: Dynamic Graph Counterfactual Explainer, a semi-supervised GCE method that uses two representation learners and a logistic regressor to find valid counterfactuals

## How to use:

Lets see an small example of how to use the framework.

### Config file

First, we need to create a config json file with the option we want to use in our experiment. In the file config/ECMLPKDD/manager_config_example_dygrace.json it is possible to find all options for each componnent of the framework.

```json
{
    "store_paths": [
        {"name": "dataset_store_path", "address": "./data/datasets/"},
        {"name": "embedder_store_path", "address": "./data/embedders/"},
        {"name": "oracle_store_path", "address": "./data/oracles/"},
        {"name": "explainer_store_path", "address": "./data/explainers/"},
        {"name": "output_store_path", "address": "./output/tree-cycles-dynamic"}
    ],
    "datasets": [
        {
            "name": "dynamic_tree_cycles", 
            "parameters": {
                "begin_time": 2000,
                "end_time": 2003,
                "num_instances_per_snapshot": 100,
                "n_nodes": 28,
                "nodes_in_cycle": 7
            } 
        }
    ],
    "oracles": [
        {
            "name": "dynamic_oracle",
            "parameters": {
                "base_oracle" : {
                    "name": "tree_cycles_custom_oracle", 
                    "parameters": {}
                },
                "first_train_timestamp": 2000
                
            }
            
        }
    ],
    "explainers": [
        {
            "name": "dygrace",
            "parameters": { 
                "in_channels": 4,
                "out_channels": 4,
                "epochs_ae": 50,
                "fold_id": 0,
                "lr": 1e-3,
                "autoencoder_name":"gae",
                "encoder_name": "gcn_encoder"
            } 
        }
    ],
    "evaluation_metrics": [ 
        {"name": "graph_edit_distance", "parameters": {}},
        {"name": "oracle_calls", "parameters": {}},
        {"name": "correctness", "parameters": {}},
        {"name": "sparsity", "parameters": {}},
        {"name": "fidelity", "parameters": {}},
        {"name": "oracle_accuracy", "parameters": {}}
    ]
}
```

Then to execute the experiment from the main the code would be something like this:

```python
from src.evaluation.dynamic_graphs.dynamic_evaluator_manager import DynamicEvaluatorManager

config_file_path = './config/ECMLPKDD/manager_config_example_dygrace.json'

print('Creating the evaluation manager.......................................................')
eval_manager = DynamicEvaluatorManager(config_file_path, K=10, run_number=0)

print('Creating the evaluators...................................................................')
eval_manager.create_evaluators()

print('Evaluating the explainers..................................................................')
eval_manager.evaluate()
```

Once the result json files are generated it is possible to use the report_dynamic_stats.py module to generate a json file with the results of the dynamic experiments.
