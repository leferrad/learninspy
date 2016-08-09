#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.utils.evaluation import ClassificationMetrics, RegressionMetrics
from learninspy.utils.fileio import get_logger
import numpy as np

logger = get_logger(name=__name__)


def test_classification_metrics():
    logger.info("Testeando métricas de evaluación para clasificación...")
    # Ejemplos para testear evaluación sobre 3 clases
    predict = [0, 1, 0, 2, 2, 1]
    labels = [0, 1, 1, 2, 1, 0]

    metrics = ClassificationMetrics(zip(predict, labels), 3)

    # Generales
    assert metrics.accuracy() == 0.5
    assert (metrics.confusion_matrix() == np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]])).all()
    # Por etiqueta
    assert metrics.precision(label=0) == metrics.precision(label=1) == metrics.precision(label=2) == 0.5
    assert (metrics.recall(label=0) == 0.5 and
            metrics.recall(label=1) == 0.3333333333333333 and
            metrics.recall(label=2) == 1.0)
    # Micro and macro
    assert metrics.precision(macro=True) == metrics.precision(macro=False) == 0.5
    assert (metrics.recall(macro=True) == 0.611111111111111 and
            metrics.recall(macro=False) == 0.5)

    # F-measure variando Beta
    assert metrics.f_measure(beta=1) == 0.5499999999999999  # F1-score, igual ponderación
    assert metrics.f_measure(beta=0.5) == 0.5188679245283019  # F0.5 score, prioriza precision en lugar de recall
    assert metrics.f_measure(beta=2) == 0.5851063829787233  # F2-score, prioriza recall en lugar de precision
    logger.info("OK")


def test_regression_metrics():
    logger.info("Testeando métricas de evaluación para regresión...")
    # Ejemplos para testear regresión
    dim = 100  # Tamaño del muestreo
    x = np.linspace(-4, 4, dim)  # Eje x
    pure = np.sinc(x)  # Función sinc (labels)
    np.random.seed(123)
    noise = np.random.uniform(0, 0.1, dim)  # Ruido para ensuciar la sinc (error en la predicción)
    signal = pure + noise  # Señal resultante del ruido aditivo (predict)
    metrics = RegressionMetrics(zip(signal, pure))

    assert np.allclose(metrics.r2(), 0.9708194315829859, rtol=1e-4)
    assert np.allclose(metrics.mse(), 0.0031164269743473839, rtol=1e-4)
    assert np.allclose(metrics.explained_variance(), 0.9943620888461356, rtol=1e-4)
    assert np.allclose(metrics.rmse(), 0.0558249673027, rtol=1e-4)
    assert np.allclose(metrics.mae(), 5.01428880051, rtol=1e-4)
    assert np.allclose(metrics.rmae(), 2.23926077099, rtol=1e-4)
    logger.info("OK")



