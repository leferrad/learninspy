__author__ = 'leferrad'

import dnn.model as mod
from dnn.optimization import OptimizerParameters
import numpy as np
import time
from sklearn import datasets

def label_to_vector(labels, nclasses):
    veclabel = []
    for l in labels:
        lab = np.zeros(nclasses)
        lab[l] = 1
        veclabel.append(lab)
    return np.array(veclabel)

parametros_red = mod.DeepLearningParams([4, 10, 5, 3], loss='CrossEntropy', activation='Tanh',
                                    dropout_ratios=[0.5, 0.5, 0.0], minimizer='Adadelta')

redneuronal = mod.NeuralNetwork(parametros_red)

iris = datasets.load_iris()
features = iris.data
labels = iris.target


t1 = time.time()
hits = redneuronal.train(features, labels, mini_batch=50, parallelism=4, epochs=3)
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits



