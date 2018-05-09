"""

This (main) script is running a grid search of various hyperparameter combinations on datasets with different hash sizes and radii.
It uses two external scripts (RFC_baseline.py and RFC_fit.py) to train the models, measure its memory and runtime.

A 5-fold cross-validation is used with an inner for loop of 5 repetitions (5 replicates).
Means and standard deviations are then calculated and outputted to a (csv) file.

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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

#track for handling .csv file at the end
for track, f in enumerate(files):
    #Load data
    data = load_svmlight_file( '../Datasets/' + f )
    X, y = data[0], data[1]
    # Remove columns that only contains zeros.
    X = X[:,X.getnnz(0)>0]

    #Set up dictionary
    results_dict = { 'radius':                           [],
                     'bit_length':                       [],
                     'depth_trees_mean':                 [],
                     'depth_trees_std':                  [],
                     'runtime_mean':                     [],
                     'runtime_std':                      [],
                     'memory_usage_std':                 [],
                     'memory_usage_mean':                [],
                     'memory_usage_max':                 [],
                     'roc_auc_score_mean':               [],
                     'roc_auc_score_std':                [],
                     'number_of_trees':                  [10,   10,   10,   10,   10,
                                                          30,   30,   30,   30,   30,
                                                          100,  100,  100,  100,  100,
                                                          300,  300,  300,  300,  300,
                                                          1000, 1000, 1000, 1000, 1000],
                     'max_features':                     [0.1,  0.3,  1.0,  3.0,  10.0,
                                                          0.1,  0.3,  1.0,  3.0,  10.0,
                                                          0.1,  0.3,  1.0,  3.0,  10.0,
                                                          0.1,  0.3,  1.0,  3.0,  10.0,
                                                          0.1,  0.3,  1.0,  3.0,  10.0] }


    kf = KFold(n_splits=5, shuffle=True)
    grid_values = { 'number_of_trees': [10, 30, 100, 300, 1000],
                    'max_features':    [int(0.10*math.sqrt(X.shape[1])), int(0.30*math.sqrt(X.shape[1])),
                                        int(1.00*math.sqrt(X.shape[1])), int(3.00*math.sqrt(X.shape[1])),
                                        int(10.00*math.sqrt(X.shape[1]))] }

    #Grid search
    for t in grid_values['number_of_trees']:
        for mf in grid_values['max_features']:

            roc_auc_scores = []
            mem_usage      = []
            runtime        = []
            depth_trees    = []

            # Five replicates
            for i in range(5):
                roc_auc_scores_repl = []
                runtime_repl        = []
                depth_trees_repl    = []
                mem_usage_repl      = []
                for train_index, test_index in kf.split(X, y):
                    X_test = X[test_index]
                    y_test = y[test_index]

                    #Measure memory on external .py file
                    train_idx = ",".join(str(x) for x in train_index)
                    test_idx = ",".join(str(x) for x in test_index)

                    while True:
                        try:
                            # peak baseline memory
                            bashCommand1 = 'python RFC_baseline.py %s %s %s %s %s' % (train_idx, t, mf, f, test_idx)
                            process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
                            mem_baseline = memory_usage(process1, interval=.05)
                            baseline = max(x for x in mem_baseline if x is not None)
                            # peak memory of algorithm - baseline
                            # same as baseline but with model.fit() and saving model
                            bashCommand2 = 'python RFC_fit.py %s %s %s %s %s' % (train_idx, t, mf, f, test_idx)
                            process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
                            mem = memory_usage(process2, interval=.05)
                            mem_max = max(x for x in mem if x is not None)-baseline
                        except:
                             continue
                        else:
                             break
                    # Import model
                    clf     = joblib.load('output/%s_RFC_model.sav' %f)
                    # Predict and obtain probabilities
                    y_proba = clf.predict_proba(X_test) ########
                    # Calculate ROC-AUC
                    roc_auc = roc_auc_score(y_test, y_proba[:,1])
                    # Obtain number of nodes
                    nodes   = np.array([estimator.tree_.node_count for estimator in clf.estimators_]).mean()
                    # Import runtime of models from output folder
                    rtime = genfromtxt('output/%s_RFC_runtime'%f, delimiter=',')

                    # Append memory usage, roc auc, runtime and number of nodes to lists.
                    mem_usage_repl.append(      mem_max )
                    roc_auc_scores_repl.append( roc_auc )
                    runtime_repl.append(        rtime   )
                    depth_trees_repl.append(    nodes   )

                # Append mean measurements from the five replicates.
                mem_usage.append(      np.mean(mem_usage_repl)      )
                depth_trees.append(    np.mean(depth_trees_repl)    )
                runtime.append(        np.mean(runtime_repl)        )
                roc_auc_scores.append( np.mean(roc_auc_scores_repl) )

            # Append results to dictionary.
            results_dict['depth_trees_mean'  ].append(round(np.mean(depth_trees), 3))
            results_dict['depth_trees_std'   ].append(round(np.std(depth_trees), 3))
            results_dict['runtime_mean'      ].append(round(np.mean(runtime), 3))
            results_dict['runtime_std'       ].append(round(np.std(runtime), 3))
            results_dict['roc_auc_score_mean'].append(round(np.mean(roc_auc_scores), 4))
            results_dict['roc_auc_score_std' ].append(round(np.std(roc_auc_scores), 4))
            results_dict['memory_usage_max'  ].append(round(np.amax([x for x in mem_usage if x is not None]), 3))
            results_dict['memory_usage_std'  ].append(round(np.std([x for x in mem_usage if x is not None]), 3))
            results_dict['memory_usage_mean' ].append(round(np.mean([x for x in mem_usage if x is not None]), 3))

            # Obtain radius and bit length directly from file name.
            if track < 18:
                results_dict['radius'].append(f[-5:-4])
                results_dict['bit_length'].append(f[-12:-8])
            else:
                results_dict['radius'].append(f[-5:-4])
                results_dict['bit_length'].append(f[-16:-8])


    df = pd.DataFrame.from_dict(results_dict)
    df.insert(loc=0, column='dataset', value=f)
    df['bit_length'].replace(regex=True,inplace=True,to_replace=r'_',value=r'')

    if track == 0:
        df.to_csv('grid_search_RF.csv')
    if track > 0:
        with open('grid_search_RF.csv', 'a') as fff:
            df.to_csv(fff, header=False)
