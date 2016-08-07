#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np


class ClassificationMetrics(object):
    """
    Métricas para evaluar en problemas de clasificación

    :param predicted_actual: list de (predicted, actual) pairs
    :param n_classes: int cantidad de clases

    >>> predict = [0, 1, 0, 2, 2, 1]
    >>> labels = [0, 1, 1, 2, 1, 0]
    >>> metrics = ClassificationMetrics(zip(predict, labels), 3)
    >>> metrics.accuracy()
    0.5
    >>> metrics.f_measure()
    0.5499999999999999
    >>> metrics.precision()
    0.5
    >>> metrics.recall()
    0.611111111111111
    >>> metrics.confusion_matrix()
    array([[1, 1, 0],
           [1, 1, 1],
           [0, 0, 1]])
    """

    # Ver http://machine-learning.tumblr.com/post/1209400132/mathematical-definitions-for-precisionrecall-for
    # Ver http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf
    def __init__(self, predicted_actual, n_classes):
        self.predicted_actual = predicted_actual
        self.tp = []
        self.fp = []
        self.fn = []
        for c in xrange(n_classes):
            self.tp.append(sum(map(lambda (p, a): p == c and a == c, predicted_actual)))
            self.fp.append(sum(map(lambda (p, a): p == c and a != c, predicted_actual)))
            self.fn.append(sum(map(lambda (p, a): p != c and a == c, predicted_actual)))
        self.n_classes = n_classes
        self.n_elem = len(predicted_actual)
        self.metrics = {'F-measure': self.f_measure, 'Accuracy': self.accuracy,
                        'Precision': self.precision, 'Recall': self.recall}

    def accuracy(self, label=None):
        """
        Calcula la exactitud de la clasificación, dada por la cantidad de aciertos sobre el total.

        :param label: int entre {0,C} para indicar sobre qué clase evaluar. Si es *None* se evalúa sobre todas.
        :return: double
        """
        if label is None:
            acc = sum(map(lambda (pre, act): pre == act, self.predicted_actual)) / float(self.n_elem)
        else:
            acc = self.tp[label] / float(self.tp[label] + self.fp[label] + self.fn[label])
        return acc

    def precision(self, label=None, macro=True):
        """
        Calcula la precisión de la clasificación, dado por la cantidad de **verdaderos positivos**
        (i.e. el número de items correctamente clasificados) dividido por el total de elementos clasificados
        para una clase dada (i.e. la suma de los verdaderos positivos y **falsos positivos**, que son los
        items incorrectamente clasificados de dicha clase). Ello se resume en la siguiente fórmula:

        :math:`P_i=\\dfrac{TP_i}{TP_i+FP_i}`

        :param label: int entre {0,n_classes - 1} para indicar sobre qué clase evaluar. Si es *None* se evalúa sobre todas.
        :param macro: bool, que indica cómo calcular el **precision** sobre todas las clases (True para que sea *macro* y False para que sea *micro*).

        Siendo C la cantidad de clases, las fórmulas son:

        :math:`P_{micro}=\\dfrac{\sum_{i=0}^{C-1} TP_i}{\sum_i TP_i+FP_i}`

        :math:`P_{macro}=\\dfrac{1}{C}\\sum_{i=0}^{C-1} \\frac{TP_i}{TP_i+FP_i}`

        :return: double
        """

        if label is None:
            if macro is True:
                p = sum([self.precision(c) for c in xrange(self.n_classes)])
                p /= float(self.n_classes)
            else:
                p = sum(self.tp) / float(sum(map(lambda (tp, fp): tp + fp, zip(self.tp, self.fp))))
        else:
            if self.tp[label] == 0.0 and self.fp[label] == 0.0:
                p = 1.0
            else:
                p = self.tp[label] / float(self.tp[label] + self.fp[label])
        return p

    def recall(self, label=None, macro=True):
        if label is None:
            if macro is True:
                r = sum([self.recall(c) for c in xrange(self.n_classes)])
                r /= float(self.n_classes)
            else:
                r = sum(self.tp) / float(sum(map(lambda (tp, fn): tp + fn, zip(self.tp, self.fn))))
        else:
            if self.tp[label] == 0.0 and self.fn[label] == 0.0:
                r = 1.0
            else:
                r = self.tp[label] / float(self.tp[label] + self.fn[label])
        return r

    def f_measure(self, beta=1, label=None, macro=True):
        ppv = self.precision(label, macro)
        tpr = self.recall(label, macro)
        if ppv == 0 and tpr == 0:
            f_score = 0.0
        else:
            f_score = (1 + beta*beta)*(ppv * tpr) / (beta*beta*ppv + tpr)
        return f_score

    def confusion_matrix(self):
        """
        Matriz de confusión resultante, donde las columnas corresponden a *predicted*
        y están ordenadas en forma ascendente por casa clase de *actual*.

        """
        conf_mat = []  # Matriz de confusion final
        for r in xrange(self.n_classes):
            pre_act = filter(lambda (p, a): a == r, self.predicted_actual)
            for c in xrange(self.n_classes):
                conf_mat.append(sum(map(lambda (p, a): p == c, pre_act)))
        return np.array(conf_mat).reshape((self.n_classes, self.n_classes))

    def evaluate(self, metric='F-measure', **kwargs):
        assert metric in self.metrics.keys(), ValueError('No se encontró la métrica '+metric+'.')
        return self.metrics[metric](**kwargs)


class RegressionMetrics(object):
    def __init__(self, predicted_actual):
        self.predicted_actual = predicted_actual
        self.n_elem = len(predicted_actual)
        self.error = map(lambda (p, a): a - p, self.predicted_actual)
        self.metrics = {'MSE': self.mse, 'RMSE': self.rmse, 'MAE': self.mae,
                        'R2': self.r2, 'ExplVar': self.explained_variance}

    def mse(self):
        return np.sum(np.square(self.error)) / float(self.n_elem)

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return np.sum(np.abs(self.error))

    def r2(self):
        # Ver https://en.wikipedia.org/wiki/Coefficient_of_determination
        mean_actual = np.mean(map(lambda (p, a): a, self.predicted_actual))
        ssres = np.sum(np.square(self.error))
        sstot = np.sum(np.square(map(lambda (p, a): a - mean_actual, self.predicted_actual)))
        return 1 - float(ssres / sstot)

    def explained_variance(self):
        var_error = np.var(self.error)
        var_actual = np.var(map(lambda (p, a): a, self.predicted_actual))
        return 1 - float(var_error / var_actual)

    def evaluate(self, metric='F-measure', **kwargs):
        assert metric in self.metrics.keys(), ValueError('No se encontró la métrica '+metric+'.')
        return self.metrics[metric](**kwargs)
