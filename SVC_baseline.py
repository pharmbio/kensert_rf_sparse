"""

This script is needed for baseline memory and will be subtracted from the SVC_fit.py memory usage.

"""

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
import sys
import numpy as np
import time
from sklearn.externals import joblib

arr = np.array(sys.argv[1].split(',')).T.astype(int)
arr2 = np.array(sys.argv[5].split(',')).T.astype(int)
data = load_svmlight_file( '../Datasets/' + sys.argv[4] )
X, y = data[0], data[1]
X = X[:,X.getnnz(0)>0]
y[y==0] = -1
X_train, y_train = X[arr], y[arr]
X_test, y_test = X[arr2], y[arr2]

clf = SVC(C=float(sys.argv[2]), gamma=float(sys.argv[3]), max_iter=250000, probability = False, class_weight="balanced")
