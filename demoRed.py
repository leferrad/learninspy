__author__ = 'leferrad'

import dnn.model as mod
from dnn.optimization import OptimizerParameters
import numpy as np
import time
from sklearn import datasets

parametros_red = mod.DeepLearningParams([4, 10, 5, 3], loss='CrossEntropy', activation='Softplus',
                                    dropout_ratios=[0.5, 0.5, 0.0], minimizer='Adadelta')

redneuronal = mod.NeuralNetwork(parametros_red)

print "Cargando base de datos ..."
data = datasets.load_iris()
features = data.data
labels = data.target
print "Size de la data: ", features.shape

print "Entrenando red neuronal ..."
t1 = time.time()
hits = redneuronal.train(features, labels, mini_batch=50, parallelism=4, epochs=5)
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits



