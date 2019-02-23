#!/usr/bin/env python3

import numpy as np

np.random.seed(20190223)

file_labels = [b"Iris-setosa", b"Iris-versicolor", b"Iris-virginica"]
data = np.loadtxt("iris.data", delimiter=",", converters={4: file_labels.index})

Itrain = np.empty((0,), dtype=np.int32)
Itest = np.empty((0,), dtype=np.int32)
NTEST = 10
    
for klass in (0, 1, 2):
    d = (data[:, -1] == klass).nonzero()[0]
    np.random.shuffle(d)
    Itrain = np.append(Itrain, d[NTEST:])
    Itest = np.append(Itest, d[:NTEST])

np.random.shuffle(Itrain)
np.random.shuffle(Itest)
Xtrain = data[Itrain, :-1]
Ytrain = data[Itrain, -1]
Xtest = data[Itest, :-1]
Ytest = data[Itest, -1]
labels = ["setosa", "versicolor", "virginica"]


import pdb; pdb.set_trace()
