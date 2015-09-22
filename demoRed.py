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

parametros_red = mod.DeepLearningParams([4, 6, 3], loss='CrossEntropy', activation='Tanh',
                                    dropout_ratios=[0.3, 0.0],minimizer='Adadelta')

redneuronal = mod.NeuralNetwork(parametros_red)

iris = datasets.load_iris()
features = iris.data
labels = iris.target


t1 = time.time()
#redneuronal.persist_layers()
hits = redneuronal.train(features, labels, mini_batch=50, epochs=10)
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits



