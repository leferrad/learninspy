#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ejemplos de uso para entrenar redes neuronales con Learninspy utilizando datos de Iris (clasificación)."""

__author__ = 'leferrad'

from learninspy.core.model import NeuralNetwork, NetworkParameters
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import LocalLabeledDataSet, load_iris
from learninspy.utils.evaluation import ClassificationMetrics
from learninspy.utils.plots import plot_fitting
from learninspy.utils.fileio import get_logger

import os

logger = get_logger(name='learninspy-demo_iris')

# Aca conviene hacer de demo:
# *Examinar diferencias en resultados con diferentes funciones de consenso
# *Explorar criterios de corte
# ** MaxIterations de 5 a 10 cambia mucho la duracion final


# -- 1) Carga de datos

logger.info("Cargando datos de Iris ...")
dataset = load_iris()
dataset = LocalLabeledDataSet(dataset)
rows, cols = dataset.shape
logger.info("Dimension de datos: %i x %i", rows, cols)

train, valid, test = dataset.split_data([0.7, 0.1, 0.2])  # Particiono en conjuntos

# -- 2) Selección de parámetros

# --- 2.a) Parámetros de red neuronal
net_params = NetworkParameters(units_layers=[4, 8, 3], dropout_ratios=[0.0, 0.0],
                               activation='ReLU', strength_l1=1e-5, strength_l2=3e-4,
                               classification=True, seed=123)

# --- 2.b) Parámetros de optimización
local_stops = [criterion['MaxIterations'](10),
               criterion['AchieveTolerance'](0.95, key='hits')]

global_stops = [criterion['MaxIterations'](30),
                criterion['AchieveTolerance'](0.95, key='hits')]

options = {'step-rate': 1.0, 'decay': 0.99, 'momentum': 0.7, 'offset': 1e-8}

optimizer_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, options=options,
                                       merge_criter='w_avg', merge_goal='hits')

#options = {'step-rate': 0.01, 'momentum': 0.8, 'momentum_type': 'nesterov'}

#opt_params = OptimizerParameters(algorithm='GD', stops=local_stops, options=options)


logger.info("Optimizacion utilizada: %s", os.linesep+str(optimizer_params))
logger.info("Configuracion usada: %s", os.linesep+str(net_params))

# -- 3) Construcción y ajuste de red neuronal

neural_net = NeuralNetwork(net_params)

logger.info("Entrenando red neuronal ...")
hits_valid = neural_net.fit(train, valid, valid_iters=1, mini_batch=10, parallelism=2, stops=global_stops,
                            optimizer_params=optimizer_params, keep_best=True, reproducible=True)
hits_test, predict = neural_net.evaluate(test, predictions=True)

logger.info("Hits en test: %12.11f", hits_test)

# --4) Evaluacion de desempeño

logger.info("Metricas de evaluación en clasificacion: ")
labels = map(lambda lp: float(lp.label), test.collect())
metrics = ClassificationMetrics(zip(predict, labels), 3)
logger.info("F1-Score: %12.11f", metrics.f_measure(beta=1))
logger.info("Precision: %12.11f", metrics.precision())
logger.info("Recall: %12.11f", metrics.recall())
logger.info("Matriz de confusion: %s", os.linesep+str(metrics.confusion_matrix()))

# --5) Ploteos

plot_fitting(neural_net)





