"""

This script is running a grid search of various hyperparameter combinations on datasets with different hash sizes and radii.
It executes external files (festlearn and festclassify, downloaded from http://lowrank.net/nikos/fest/) to train the models and predict on test set.
Memory usage and runtime are measured for festlearn; and prediction file is read to calculate ROC AUC.

A 5-fold cross-validation is used with an inner for loop of 5 repetitions (5 replicates).
Means and standard deviations are then calculated and outputted to a (csv) file.

"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file
import math
from sklearn.datasets import load_svmlight_file
import time
import subprocess
from memory_profiler import memory_usage


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

# For keeping track of progression
dataset = 1
for track, f in enumerate(files):
    # For keeping track of progression
    runs = 1

    data = load_svmlight_file( 'Datasets/' + f )
    X, y = data[0], data[1]
    X = X[:,X.getnnz(0)>0]


    results_dict = { 'radius':                  [],
                     'bit_length':              [],
                     'nodes_trees_mean':        [],
                     'nodes_trees_std':         [],
                     'runtime_mean':            [],
                     'runtime_std':             [],
                     'memory_usage_std':        [],
                     'memory_usage_mean':       [],
                     'memory_usage_max':        [],
                     'roc_auc_score_mean':      [],
                     'roc_auc_score_std':       [],
                     'number_of_trees':         [10, 10, 10, 10, 10,
                                                 30, 30, 30, 30, 30,
                                                 100, 100, 100, 100, 100,
                                                 300, 300, 300, 300, 300,
                                                 1000, 1000, 1000, 1000, 1000],
                     'max_features':            [0.1,  0.3,  1.0,  3.0,  10.0,
                                                 0.1,  0.3,  1.0,  3.0,  10.0,
                                                 0.1,  0.3,  1.0,  3.0,  10.0,
                                                 0.1,  0.3,  1.0,  3.0,  10.0,
                                                 0.1,  0.3,  1.0,  3.0,  10.0]  }


    kf = KFold(n_splits=5, shuffle=True)
    grid_values = { 'number_of_trees': [10,   30,   100,  300,  1000],
                    'max_features':    [0.10, 0.30, 1.00, 3.00, 10.00] }


    ###output training and test data sets
    #y_true = []
    #i = 1


    for trees in grid_values['number_of_trees']:
        for features in grid_values['max_features']:
            roc_auc_scores = []
            mem_usage = []
            runtime = []
            nodes_trees = []
            for a in range(5):
                roc_auc_scores_repl = []
                runtime_repl = []
                nodes_trees_repl     = []
                mem_usage_repl = []

                i = 1
                c = 0
                for train_index, test_index in kf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    neg_weight = (y_train == 1).sum()/(y_train == -1).sum()

                    dump_svmlight_file(X_train, y_train, f='output/train%i' % i)
                    dump_svmlight_file(X_test, y_test, f='output/test%i' % i)

                    bashCommand = './festlearn -c 3 -n %s -p %s -t %s output/train%i output/model%i' % (neg_weight,
                                                                                                        features,
                                                                                                        trees,
                                                                                                        i,
                                                                                                        i)
                    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                    # This while loop makes sure that memory usage is obtained for every run;
                    # if it fails, it runs again
                    while True:
                        try:
                            tic = time.time()
                            # peak memory usage; also executes "process" which outputs the model
                            mem_max = max(memory_usage(process, interval=.05))
                            toc = time.time()
                            # Make predictions on test set
                            bashCommand2 = "./festclassify output/test%i output/model%i output/preds%i" % (i, i, i)
                            process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
                            output, error = process2.communicate()
                        except:
                            continue
                        else:
                            break
                    # Read predictions file to calculate ROC AUC
                    with open('output/preds%i' % (i)) as ff:
                        lines = ff.read().splitlines()
                        lines = [float(i) for i in lines]
                        lines = np.array(lines)

                    # Append memory usage, roc auc, runtime and number of nodes to lists.
                    roc_auc_scores_repl.append(roc_auc_score(y_test, lines))
                    mem_usage_repl.append(mem_max)
                    runtime_repl.append(toc-tic)

                    # Read model's file to calculate tree nodes
                    with open('output/model%i' % (i)) as ff:
                        lines = ff.read().splitlines()
                        lines = lines[5:]
                        nodes = []
                        for j in lines:
                            nodes_temp = len([s for s in j.split() if s.isdigit()])
                            nodes.append(nodes_temp)

                    nodes_trees_repl.append(np.mean(nodes))
                    c += 1
                    i += 1

                    if runs % 5 == 0:
                        print('Current dataset: ' + str(runs) + "/625 runs completed")
                    runs += 1
                # Append mean measurements from the five replicates.
                mem_usage.append(np.mean(mem_usage_repl))
                nodes_trees.append(np.mean(nodes_trees_repl))
                runtime.append(np.mean(runtime_repl))
                roc_auc_scores.append(np.mean(roc_auc_scores_repl))

            results_dict['nodes_trees_mean'].append(   round(np.mean(nodes_trees), 3)    )
            results_dict['nodes_trees_std'].append(    round(np.std(nodes_trees), 3)     )
            results_dict['runtime_mean'].append(       round(np.mean(runtime), 3)        )
            results_dict['runtime_std'].append(        round(np.std(runtime), 3)         )
            results_dict['roc_auc_score_mean'].append( round(np.mean(roc_auc_scores), 4) )
            results_dict['roc_auc_score_std'].append(  round(np.std(roc_auc_scores), 4)  )
            results_dict['memory_usage_max'].append(   round(np.amax([x for x in mem_usage if x is not None]), 3))
            results_dict['memory_usage_std'].append(   round(np.std([x for x in mem_usage if x is not None]), 3))
            results_dict['memory_usage_mean'].append(  round(np.mean([x for x in mem_usage if x is not None]), 3))
            if track < 18:
                results_dict['radius'].append(f[-5:-4])
                results_dict['bit_length'].append(f[-12:-8])
            else:
                results_dict['radius'].append(f[-11:-10])
                results_dict['bit_length'].append(f[-22:-14])

    print(str(dataset) + "/21 datasets completed")
    dataset += 1

    df = pd.DataFrame.from_dict(results_dict)
    df.insert(loc=0, column='dataset', value=f)
    df['bit_length'].replace(regex=True,inplace=True,to_replace=r'_',value=r'')
    if track == 0:
        df.to_csv('grid_search_FEST.csv')
    if track > 0:
        with open('grid_search_FEST.csv', 'a') as fff:
            df.to_csv(fff, header=False)
