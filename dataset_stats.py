from src.dataset.dataset_factory import DatasetFactory
from src.data_analysis.data_analyzer import DataAnalyzer

import numpy as np
# data_store_path = './data/datasets/'
# dtan = DataAnalyzer('./output', './stats')

data_store_path = './data/datasets/'
dtan = DataAnalyzer('./output/steel/BBBP', './stats/btc_otc/')


datasets =[
    {
             "name": "btc_otc", 
            "parameters": {
                "begin_time": 2011,
                "end_time": 2015,
                "filter_min_graphs": 4,
                "number_of_communities": 4
            } 
        }
]

dtan.get_datasets_stats(datasets, data_store_path)