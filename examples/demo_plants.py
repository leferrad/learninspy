#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ejemplos de uso para entrenar redes neuronales con Learninspy
utilizando datos de Combined Cycle Power Plant (regresión)."""

__author__ = 'leferrad'

from learninspy.core.model import NeuralNetwork, NetworkParameters
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import LocalLabeledDataSet, load_ccpp
from learninspy.utils.evaluation import RegressionMetrics
from learninspy.utils.plots import plot_fitting
from learninspy.utils.fileio import get_logger

import os

logger = get_logger(name='learninspy-demo_ccpp')

# -- 1.a) Carga de datos

logger.info("Cargando datos de Combined Cycle Power Plant ...")
dataset = load_ccpp()
dataset = LocalLabeledDataSet(dataset)
rows, cols = dataset.shape
logger.info("Dimension de datos: %i x %i", rows, cols)

train, valid, test = dataset.split_data([0.5, 0.3, 0.2])  # Particiono en conjuntos

# -- 1.b) Normalización
"""
std = StandardScaler()
std.fit(train)
train = std.transform(train)
valid = std.transform(valid)
test = std.transform(test)
"""
# -- 2) Selección de parámetros

# --- 2.a) Parámetros de red neuronal
net_params = NetworkParameters(units_layers=[4, 30, 1], dropout_ratios=[0.0, 0.0],
                               activation='ReLU', strength_l1=5e-7, strength_l2=3e-4,
                               classification=False, seed=123)

# --- 2.b) Parámetros de optimización
local_stops = [criterion['MaxIterations'](30),
               criterion['AchieveTolerance'](0.95, key='hits')]

global_stops = [criterion['MaxIterations'](20),
                criterion['AchieveTolerance'](0.95, key='hits')]

options = {'step-rate': 1.0, 'decay': 0.995, 'momentum': 0.7, 'offset': 1e-8}

optimizer_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, options=options,
                                       merge_criter='w_avg', merge_goal='cost')

logger.info("Optimizacion utilizada: %s", os.linesep+str(optimizer_params))
logger.info("Configuracion usada: %s", os.linesep+str(net_params))

# -- 3) Construcción y ajuste de red neuronal

neural_net = NeuralNetwork(net_params)

logger.info("Entrenando red neuronal ...")
hits_valid = neural_net.fit(train, valid, valid_iters=1, mini_batch=20, parallelism=0,
                            stops=global_stops, optimizer_params=optimizer_params, measure='R2',
                            keep_best=True, reproducible=False)
hits_test, predict = neural_net.evaluate(test, predictions=True)

logger.info("Hits en test: %12.11f", hits_test)

# --4) Evaluacion de desempeño

logger.info("Metricas de evaluación en clasificacion: ")
labels = map(lambda lp: float(lp.label), test.collect())
metrics = RegressionMetrics(zip(predict, labels))
print "MSE: ", metrics.mse()
print "RMSE: ", metrics.rmse()
print "MAE: ", metrics.mae()
print "RMAE: ", metrics.rmae()
print "R-cuadrado: ", metrics.r2()
print "Explained Variance: ", metrics.explained_variance()
print zip(predict, labels)[:10]

# --5) Ploteos

plot_fitting(neural_net)
