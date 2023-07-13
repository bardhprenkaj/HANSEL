import glob
import os
import re
import sys
import json
import numpy as np
import pandas as pd

from src.evaluation.evaluation_metric_correctness import CorrectnessMetric
from src.evaluation.evaluation_metric_fidelity import FidelityMetric
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.evaluation.evaluation_metric_oracle_accuracy import \
    OracleAccuracyMetric
from src.evaluation.evaluation_metric_sparsity import SparsityMetric

def update_dictionary(dict1, dict2):
    for key, value in dict2.items():
        for key1, value1 in value.items():
            flattened_val = []
            for vals in value1:
                if type(vals) == list:
                    flattened_val += vals
                else:
                    flattened_val.append(vals)
            if key1 in dict1:
                dict1[key][key1] += flattened_val
            else:
                dict1[key][key1] = flattened_val
            
if __name__ == '__main__':
    
    output_folder = sys.argv[1]
    k = int(sys.argv[2])
    
    metrics = [GraphEditDistanceMetric(), OracleAccuracyMetric(), SparsityMetric(), FidelityMetric(), CorrectnessMetric()]
    
    if os.path.exists(output_folder):
        files = [x[0] for x in os.walk(output_folder)][1:]
        df = pd.DataFrame(files, columns=['out_path'])
        df['year'] = df.out_path.apply(lambda path : re.findall('(\d{4})', path)[0])
        
        report = {year: {} for year in df.year.unique()}

        for group in df.groupby(by='year'):
            onlyfiles = np.array([glob.glob(f'{filepath}/*') for filepath in group[1].out_path.values]).flatten()
            for file in onlyfiles:
                with open(file, 'r') as f:
                    reports = json.load(f)
                    for i in range(1, k+1):
                        for metric in metrics:
                            curr_metric = reports.get(f'{metric._name}@{i}', [])
                            update_dictionary(report, {group[0] : {f'{metric._name}@{i}': curr_metric}})
        
        for year in report.keys():
            for metric in report[year]:
                report[year][metric] = f'{np.mean(report[year][metric])} \pm {np.std(report[year][metric])}'
        
        with open('yearly_metric_reports.json', 'w') as f:
            json.dump(report, f)