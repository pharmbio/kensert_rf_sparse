"""

This script takes arguments from main script (RFC_grid_search.py) and trains a model.

The script's main purpose is to obtain a relatively accurate measurement of memory usage,
that is also comparable with the measuring of the FEST algorithm's memory usage.

function "memory_usage" from main script is used.

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
import sys
import numpy as np
import time
from sklearn.externals import joblib

arr = np.array(sys.argv[1].split(',')).T.astype(int)
arr2 = np.array(sys.argv[5].split(',')).T.astype(int)
data = load_svmlight_file( 'Datasets/' + sys.argv[4] )
X, y = data[0], data[1]
X = X[:,X.getnnz(0)>0]
y[y==0] = -1
X_train, y_train = X[arr], y[arr]
X_test, y_test = X[arr2], y[arr2]

clf = RandomForestClassifier(criterion='entropy', n_estimators=int(sys.argv[2]), max_features=int(sys.argv[3]), n_jobs=-1, class_weight="balanced")

tic = time.time()
clf.fit(X_train, y_train)
toc = time.time()

joblib.dump(clf, 'output/%s_RFC_model.sav' % sys.argv[4])

rtime   = toc-tic
lst = [rtime]

np.savetxt('output/%s_RFC_runtime' % sys.argv[4], lst, delimiter=',', fmt='%s')
