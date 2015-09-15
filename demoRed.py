__author__ = 'leferrad'

import dnn.model as mod
import numpy as np
import time


parametros = mod.DeepLearningParams([200, 100, 50, 10], loss='MSE', activation='LeakyReLU',
                                    dropout_ratios=[0.5,0.5,0.5])
redneuronal = mod.NeuralNetwork(parametros)

entrada = np.asarray(parametros.rng.uniform(low=-3.0, high=3.0, size=(200, 1)), dtype=np.dtype(float))
#entradaRDD = sc.parallelize(entrada)

t1 = time.time()
#redneuronal.persist_layers()
cost, gradient = redneuronal.cost(entrada, np.ones(10))
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'cost: ', cost

nabla_w, nabla_b = gradient
for i in xrange(len(nabla_b)):
    print 'W', nabla_w[i].shape()
    print 'b', nabla_b[i].shape()
    print '--'


