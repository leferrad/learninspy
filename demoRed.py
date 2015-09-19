__author__ = 'leferrad'

import dnn.model as mod
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

parametros = mod.DeepLearningParams([4, 10, 5, 3], loss='CrossEntropy', activation='LeakyReLU',
                                    dropout_ratios=[0.5, 0.5, 0.0])
redneuronal = mod.NeuralNetwork(parametros)

iris = datasets.load_iris()
features = iris.data
labels = iris.target

t1 = time.time()
#redneuronal.persist_layers()
hits = redneuronal.train(features, labels)
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits



