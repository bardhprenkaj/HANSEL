
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

HANSEL is an open-source framework for Evaluating Graph Counterfactual Explanation Methods under data changes (i.e., data distribution drifts). Our main goal is to create a generic platform that allows the researchers to speed up the process of developing and testing new Timed-Graph Counterfactual Explanation Methods.


## Citation Request:

Please cite our paper if you use HANSEL in your experiments:

Prenkaj, B.; Villaiz ́an-Vallelado, M.; Leemann, T.; and Kasneci, G. 2023. Unifying Evolution, Explanation, and Discernment: A Generative Approach for Dynamic Graph Counterfactuals. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3637528.3671831

```latex:
@inproceedings{prenkaj2024gracie,
  title={Unifying Evolution, Explanation, and Discernment: A Generative Approach for Dynamic Graph Counterfactuals},
  author={Bardh Prenkaj and Mario Villaizán-Vallelado and Tobias Leemann and Gjergji Kasneci},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24)},
  year={2024}
}
```

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

The image is based on `tensorflow/tensorflow:latest-gpu` and it's GPU ready. To set up the container, we recommend you run:

```
docker-compose run gretel
```

For simplicity, we provide several **makefile** rules for easy interaction with the Docker interface:

 * `make docker` - builds the development image from scratch
 * `make pull` - pull the development image
 * `make push` - push the development image
 * `make demo` - run the demo in the development image.

## Resources provided with the Framework:

### Datasets:

* **DynTree-Cycles**: Synthetic data set where each instance is a graph. The instance can be either a tree or a tree with several cycle patterns connected to the main graph by one edge

* **DBLP-Coauthors**: It contains graphs where the vertices represent authors, and the connections are co-authorship relationships between two authors.

* **BTC-Alpha** and **BTC-OTC**: It contains who-trust-whom networks of traderson the Bitcoin Alpha and Bitcoin OTC platforms

* **Bonanza**:  This is similar to eBay and Amazon Marketplace in that users create an account to buy or sell various goods. After a buyer purchases a product from a seller, both can provide a rating about the other along with a short comment.

### Oracles:

* **KNN**

* **SVM**

* **GCN**

* **CustomOracle**

### Explainers:

* **DyGRACE**: Dynamic Graph Counterfactual Explainer, a semi-supervised GCE method that uses two representation learners and a logistic regressor to find valid counterfactuals

* **GRACIE**: Graph Recalibration and Adaptive Counterfactual Inspection and Explanation

## How to use:

Let's see a small example of how to use the framework.

### Config file

First, we must create a config JSON file with the option we want to use in our experiment. In the file config/best_models/tree-cycles/gracie/0.json, it is possible to find all options for each component of the framework.

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
            "name": "gracie",
            "parameters": { 
                "epochs_ae": 200,
                "fold_id": 0,
                "lr": 1e-3,
                "top_k_cf": 50,
                "batch_size": 64,
                "kl_weight": 0.5,
                "in_dim": 4,
                "decoder_dims": 2,
                "replace_rate": 0.1,
                "mask_rate": 0.3,
                "lambda": 0.5,
                "encoder": {
                    "name": "var_gcn_encoder",
                    "parameters": {
                        "input_dim": 4,
                        "out_dim": 2
                    }
                },
                "decoder": {
                    "name": "gat_decoder",
                    "parameters": {
                        "input_dim": 2,
                        "hidden_dim": 3,
                        "out_dim": 4,
                        "num_layers": 2,
                        "nhead": 16,
                        "nhead_out": 1,
                        "negative_slope": 0.2                    
                    }
                },
                "weight_schedulers": [
                    {
                        "name": "no_decay",
                        "parameters" : {
                            "init_weight": 1
                        }
                    },
                    {
                        "name": "no_decay",
                        "parameters": {
                            "init_weight": 0.5
                        }
                    }
                ]
            } 
        }
    ],
    "evaluation_metrics": [ 
        {"name": "graph_edit_distance", "parameters": {}},
        {"name": "oracle_calls", "parameters": {}},
        {"name": "correctness", "parameters": {}},
        {"name": "oracle_accuracy", "parameters": {}}
    ]
}
```

Then to execute the experiment from the main the code would be something like this:

```python
from src.evaluation.dynamic_graphs.dynamic_evaluator_manager import DynamicEvaluatorManager

config_file_path = './config/best_models/tree-cycles/gracie/0.json'

print('Creating the evaluation manager.......................................................')
eval_manager = DynamicEvaluatorManager(config_file_path, K=10, run_number=0)

print('Creating the evaluators...................................................................')
eval_manager.create_evaluators()

print('Evaluating the explainers..................................................................')
eval_manager.evaluate()
```

Once the result JSON files are generated, it is possible to use the report_dynamic_stats.py module to generate a JSON file with the results of the dynamic experiments.
