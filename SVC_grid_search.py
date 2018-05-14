"""

This (main) script is running a grid search of various hyperparameter combinations on datasets with different hash sizes and radii.
It uses two external scripts (SVC_baseline.py and SVC_fit.py) to train the models, measure its memory and runtime.

A 5-fold cross-validation is used with an inner for loop of 5 repetitions (5 replicates).
Means and standard deviations are then calculated and outputted to a (csv) file.

"""


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import KFold
from memory_profiler import memory_usage
import math
import sys
import subprocess
import time
from numpy import genfromtxt
from sklearn.externals import joblib

# Files/Datasets (in svmlight-/libsvm-format),
# replace "dataset" with the appropriate file name (outputs from gen_morgan.py).
files = [
'dataset_svmlight-format_hashed_128_r_1.csv',
'dataset_svmlight-format_hashed_256_r_1.csv',
'dataset_svmlight-format_hashed_512_r_1.csv',
'dataset_svmlight-format_hashed_1024_r_1.csv',
'dataset_svmlight-format_hashed_2048_r_1.csv',
'dataset_svmlight-format_hashed_4096_r_1.csv',
'dataset_svmlight-format_hashed_128_r_2.csv',
'dataset_svmlight-format_hashed_256_r_2.csv',
'dataset_svmlight-format_hashed_512_r_2.csv',
'dataset_svmlight-format_hashed_1024_r_2.csv',
'dataset_svmlight-format_hashed_2048_r_2.csv',
'dataset_svmlight-format_hashed_4096_r_2.csv',
'dataset_svmlight-format_hashed_128_r_3.csv',
'dataset_svmlight-format_hashed_256_r_3.csv',
'dataset_svmlight-format_hashed_512_r_3.csv',
'dataset_svmlight-format_hashed_1024_r_3.csv',
'dataset_svmlight-format_hashed_2048_r_3.csv',
'dataset_svmlight-format_hashed_4096_r_3.csv',
'dataset_svmlight-format_unhashed_r_1.csv',
'dataset_svmlight-format_unhashed_r_2.csv',
'dataset_svmlight-format_unhashed_r_3.csv'
]

grid_values = { 'gamma': np.array([0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]),
                'C':    np.array([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]) }
kf = KFold(n_splits=5, shuffle=True)

# For keeping track of progression
dataset = 1

for track,f in enumerate(files):
    # For keeping track of progression
    runs = 1

    #Load data
    data = load_svmlight_file( 'Datasets/' + f )
    X, y = data[0], data[1]
    X = X[:,X.getnnz(0)>0]

    #Set up results_dictionary
    results_dict = { 'radius':                           [],
            'bit_length':                           [],
            'runtime_mean':                        [],
            'runtime_std':                         [],
            'memory_usage_std':                       [],
            'memory_usage_mean':                      [],
            'memory_usage_max':                       [],
            'roc_auc_score_mean':        [],
            'roc_auc_score_std':         [],
            'C':                                np.repeat(grid_values['C'], 11),
            'gamma':                            np.tile(grid_values['gamma'], 11) }


    #Grid search
    for cost in grid_values['C']:
        for gamma in grid_values['gamma']:

            roc_auc_scores = []
            mem_usage = []
            runtime = []


            for i in range(5):
                roc_auc_scores_repl = []
                runtime_repl = []
                mem_usage_repl = []
                for train_index, test_index in kf.split(X, y):
                    X_test = X[test_index]
                    y_test = y[test_index]

                    #Measure memory on external .py file
                    train_idx = ",".join(str(x) for x in train_index)
                    test_idx = ",".join(str(x) for x in test_index)
                    while True:
                        try:
                            # peak baseline memory
                            bashCommand1 = 'python SVC_baseline.py %s %s %s %s %s' % (train_idx, cost, gamma, f, test_idx)
                            process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
                            mem_baseline = memory_usage(process1, interval=.05)
                            baseline = max(x for x in mem_baseline if x is not None)
                            # peak memory of algorithm - baseline
                            # same as baseline but with model.fit() and saving model
                            bashCommand2 = 'python SVC_fit.py %s %s %s %s %s' % (train_idx, cost, gamma, f, test_idx)
                            process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
                            mem = memory_usage(process2, interval=.05)
                            mem_max = max(x for x in mem if x is not None)-baseline
                        except:
                             continue
                        else:
                             break

                    clf = joblib.load('output/%s_SVC_model.sav' %f)

                    # Predict and obtain probabilities
                    y_proba = clf.decision_function(X_test)
                    # Calculate ROC-AUC
                    roc_auc = roc_auc_score(y_test, y_proba)
                    # Import runtime of models from output folder
                    time = genfromtxt('output/%s_SVC_runtime'%f, delimiter=',')
                    # Append memory usage, roc auc and runtime to lists.
                    mem_usage_repl.append(mem_max)
                    roc_auc_scores_repl.append(roc_auc)
                    runtime_repl.append(time)

                    print('Current dataset: ' + str(runs) + "/500 runs completed")
                    runs += 1

                # Append mean measurements from the five replicates.
                mem_usage.append(np.mean(mem_usage_repl))
                runtime.append(np.mean(runtime_repl))
                roc_auc_scores.append(np.mean(roc_auc_scores_repl))

            # Append results to dictionary.
            results_dict['runtime_mean'].append(round(np.mean(runtime), 3))
            results_dict['runtime_std'].append(round(np.std(runtime), 3))
            results_dict['roc_auc_score_mean'].append(round(np.mean(roc_auc_scores), 4))
            results_dict['roc_auc_score_std'].append(round(np.std(roc_auc_scores), 4))
            results_dict['memory_usage_max'].append(round(np.amax([x for x in mem_usage if x is not None]), 3))
            results_dict['memory_usage_std'].append(round(np.std([x for x in mem_usage if x is not None]), 3))
            results_dict['memory_usage_mean'].append(round(np.mean([x for x in mem_usage if x is not None]), 3))
            if track < 18:
                results_dict['radius'].append(f[-5:-4])
                results_dict['bit_length'].append(f[-12:-8])
            else:
                results_dict['radius'].append(f[-5:-4])
                results_dict['bit_length'].append(f[-16:-8])

            print(str(dataset) + "/21 datasets completed")
            dataset += 1

    df = pd.DataFrame.from_dict(results_dict)
    df.insert(loc=0, column='dataset', value=f)
    df['bit_length'].replace(regex=True,inplace=True,to_replace=r'_',value=r'')
    if track == 0:
        df.to_csv('grid_search_SVC.csv')
    if track > 0:
        with open('grid_search_SVC.csv', 'a') as fff:
            df.to_csv(fff, header=False)
