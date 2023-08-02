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
            
if __name__ == '__main__':
    
    output_folder = sys.argv[1]
    k = int(sys.argv[2])
    
    metrics = [GraphEditDistanceMetric(), OracleAccuracyMetric(), SparsityMetric(), FidelityMetric(), CorrectnessMetric()]
    
    if os.path.exists(output_folder):
        files = [x[0] for x in os.walk(output_folder)][1:]
        df = pd.DataFrame(files, columns=['out_path'])
        df['year'] = df.out_path.apply(lambda path : re.findall('(\d{4})', path)[0])
        
        report = {year: {f'{metric._name}@{i}': [] for i in range(1,k+1) for metric in metrics}  for year in df.year.unique()}
        for year in report.keys():
            report[year].setdefault("runtime", [])

        for group in df.groupby(by='year'):
            onlyfiles = np.array([glob.glob(f'{filepath}/*') for filepath in group[1].out_path.values]).flatten()
            for file in onlyfiles:
                with open(file, 'r') as f:
                    reports = json.load(f)
                    for i in range(1, k+1):
                        for metric in metrics:
                            curr_metric = reports.get(f'{metric._name}@{i}', [])
                            if f'{metric._name}'.lower() in ['graph_edit_distance', 'sparsity']:
                                curr_metric = np.mean(curr_metric, axis=1).tolist()
                            if type(curr_metric) == list:
                                report[group[0]][f'{metric._name}@{i}'].append(curr_metric)
                            else:
                                report[group[0]][f'{metric._name}@{i}'].append([float(curr_metric)] * len(report[group[0]][f'{metric._name}@{i}']))
                            
                    runtime = reports.get('runtime', [])
                    if type(runtime) == list:
                        report[group[0]]['runtime'].append(runtime)
                    else:
                        report[group[0]]['runtime'].append([float(runtime)] * len(report[group[0]]['runtime']))
        
        for year in report.keys():
            for metric in report[year]:
                report[year][metric] = np.mean(report[year][metric], axis=0)
                report[year][metric] = f'{np.mean(report[year][metric]):.2f}\\pm {np.std(report[year][metric]):.3f}'
             
            if type(report[year]['runtime']) != str:
                report[year]['runtime'] = np.array(report[year]['runtime']).astype(np.float64)
                report[year]['runtime'] = f'{np.mean(report[year]["runtime"]):.2f}^'+"{"+f'\\pm {np.std(report[year]["runtime"]):.3f}'+"}"
        
        with open('yearly_metric_reports.json', 'w') as f:
            json.dump(report, f)