#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Para conocer el comportamiento del modelo construido en la tarea asignada, se establecen métricas para medir
su desempeño y con ello ajustar el mismo para mejorar sus resultados.
Dichas métricas son específicas del tipo de problema tratado, por lo que se distinguen
para las tareas soportadas en el modelado: **clasificación** y **regresión**.
"""

__author__ = 'leferrad'

import numpy as np


class ClassificationMetrics(object):
    """
    Métricas para evaluar el desempeño de un modelo en problemas de clasificación.

    Basadas en la lista de métricas presentadas en la publicación de Sokolova et.al. [sokolova2009systematic]_.


    :param predicted_actual: list de tuples (predicted, actual)
    :param n_classes: int, cantidad de clases tratadas en la tarea de clasificación.

    >>> predict = [0, 1, 0, 2, 2, 1]
    >>> labels = [0, 1, 1, 2, 1, 0]
    >>> metrics = ClassificationMetrics(zip(predict, labels), 3)
    >>> metrics.measures.keys()
    ['Recall', 'F-measure', 'Precision', 'Accuracy']
    >>> metrics.accuracy()
    0.5
    >>> metrics.f_measure()
    0.5499999999999999
    >>> metrics.precision()
    0.5
    >>> metrics.evaluate('Recall')
    0.611111111111111
    >>> metrics.confusion_matrix()
    array([[1, 1, 0],
           [1, 1, 1],
           [0, 0, 1]])

    **Referencias**:

    .. [sokolova2009systematic] Sokolova, M., & Lapalme, G. (2009).
        A systematic analysis of performance measures for classification tasks.
        Information Processing & Management, 45(4), 427-437.
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
        self.measures = {'F-measure': self.f_measure, 'Accuracy': self.accuracy,
                         'Precision': self.precision, 'Recall': self.recall}

    def accuracy(self, label=None):
        """
        Calcula la exactitud de la clasificación, dada por la cantidad de aciertos sobre el total.

        Siendo C la cantidad de clases, la fórmula para calcular dicho valor es:

        :math:`ACC=\\dfrac{1}{C}\displaystyle\sum\limits_{i=0}^{C-1} \\frac{TP_i + TN_i}{TP_i+FN_i+FP_i+TN_i}`

        :param label: int entre {0,C} para indicar sobre qué clase evaluar. Si es *None* se evalúa sobre todas.
        :return: float, que puede variar entre 0 (peor)  y 1 (mejor).
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

        Siendo C la cantidad de clases, las fórmulas para el micro- y macro-averaging son:

        :math:`P_{\\mu}=\\dfrac{\sum_{i=0}^{C-1} TP_i}{\sum_i TP_i+FP_i}, \quad
        P_{M}=\\dfrac{1}{C}\displaystyle\sum\limits_{i=0}^{C-1} \\frac{TP_i}{TP_i+FP_i}`

        :param label: int entre {0, C - 1} para indicar sobre qué clase evaluar. Si es *None* se evalúa sobre todas.
        :param macro: bool, que indica cómo calcular el **precision** sobre todas las clases
         (True para que sea *macro* y False para que sea *micro*).
        :return: float, que puede variar entre 0 (peor)  y 1 (mejor).
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
        """
        Calcula la exhaustividad de la clasificación, dado por la cantidad de **verdaderos positivos**
        (i.e. el número de items correctamente clasificados) dividido por el total de elementos que realmente
        pertenecen a la clase en cuestión (i.e. la suma de los verdaderos positivos y **falsos negativos**, que son los
        items incorrectamente no clasificados como dicha clase). Ello se resume en la siguiente fórmula:

        :math:`R_i=\\dfrac{TP_i}{TP_i+FN_i}`

        Siendo C la cantidad de clases, las fórmulas para el micro- y macro-averaging son:

        :math:`R_{\\mu}=\\dfrac{\sum_{i=0}^{C-1} TP_i}{\sum_i TP_i+FN_i}, \quad
        R_{M}=\\dfrac{1}{C}\displaystyle\sum\limits_{i=0}^{C-1} \\frac{TP_i}{TP_i+FN_i}`

        :param label: int entre {0, C - 1} para indicar sobre qué clase evaluar. Si es *None* se evalúa sobre todas.
        :param macro: bool, que indica cómo calcular el **recall** sobre todas las clases
         (True para que sea *macro* y False para que sea *micro*).
        :return: float, que puede variar entre 0 (peor)  y 1 (mejor).
        """
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
        """
        Calcula el *F-measure* de la clasificación, el cual combina las medidas de *precision* y *recall* mediante
        una media armónica de ambos. Dicho balance es ajustado por un parámetro :math:`\\beta`, y un caso
        muy utilizado de esta medida es el *F1-score* donde se pondera igual a ambas medidas con :math:`\\beta = 1`.

        :math:`F(\\beta)=(1+\\beta)(\\dfrac{PR}{\\beta^2 P + R}), \quad F_1=\\dfrac{2PR}{P + R}`

        Siendo C la cantidad de clases, las fórmulas para el micro- y macro-averaging son:

        :math:`F_{\\mu}(\\beta)=(1+\\beta)(\\dfrac{P_{\\mu}R_{\\mu}}{\\beta^2 P_{\\mu} + R_{\\mu}}), \quad
        F_{M}(\\beta)=(1+\\beta)(\\dfrac{P_{M}R_{M}}{\\beta^2 P_{M} + R_{M}})`


        :param beta: float, parámetro :math:`\\beta` que determina el balance entre *precision* y *recall*.
         Si :math:`\\beta < 1` se prioriza el *precision*, mientras que con :math:`\\beta > 1` se favorece al *recall*.
        :param label: int entre {0, C - 1} para indicar sobre qué clase evaluar. Si es *None* se evalúa sobre todas.
        :param macro: bool, que indica cómo calcular el **F-measure** sobre todas las clases
         (True para que sea *macro* y False para que sea *micro*).
        :return: float, que puede variar entre 0 (peor)  y 1 (mejor).
        """
        ppv = self.precision(label, macro)
        tpr = self.recall(label, macro)
        if ppv == 0 and tpr == 0:
            f_score = 0.0
        else:
            f_score = (1 + beta*beta)*(ppv * tpr) / (beta*beta*ppv + tpr)
        return f_score

    def confusion_matrix(self):
        """
        Matriz de confusión resultante, donde las columnas corresponden a los valores de *predicted*
        y están ordenadas en forma ascendente por cada clase de *actual*.

        Para realizar un ploteo del resultado, se puede recurrir a la función
        :func:`~learninspy.utils.plots.plot_confusion_matrix`.

        :return: numpy.ndarray
        """
        conf_mat = []  # Matriz de confusion final
        for r in xrange(self.n_classes):
            pre_act = filter(lambda (p, a): a == r, self.predicted_actual)
            for c in xrange(self.n_classes):
                conf_mat.append(sum(map(lambda (p, a): p == c, pre_act)))
        return np.array(conf_mat).reshape((self.n_classes, self.n_classes))

    def evaluate(self, measure='F-measure', **kwargs):
        """
        Aplica alguna de las medidas implementadas, las cuales se encuentran registradas en el dict
        *self.measures*. Esta función resulta práctica para parametrizar fácilmente la medida a utilizar
        durante el ajuste de un modelo.

        :param measure: string, key de alguna medida implementada.
        :param kwargs: se pueden incluir otros parámetros propios de la medida a utilizar (e.g. *beta* para
         *F-measure*, o *micro / macro* para aquellas que lo soporten).
        :return: float
        """
        assert measure in self.measures.keys(), ValueError('No se encontró la medida '+measure+'.')
        return self.measures[measure](**kwargs)


class RegressionMetrics(object):
    """

    Métricas para evaluar el desempeño de un modelo en problemas de regresión.

    :param predicted_actual: list de tuples (predicted, actual)

    >>> predict = [0.5, 1.1, 1.5, 2.0, 3.5, 5.2]
    >>> labels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    >>> metrics = RegressionMetrics(zip(predict, labels))
    >>> metrics.measures.keys()
    ['ExplVar', 'MSE', 'MAE', 'R2', 'RMSE']
    >>> metrics.mae()
    2.3000000000000003
    >>> metrics.mse()
    0.25833333333333336
    >>> metrics.evaluate('RMSE')
    0.50826502273256358
    >>> metrics.r2()
    0.8980821917808219
    >>> metrics.explained_variance()
    0.9297534246575342
    """
    def __init__(self, predicted_actual):
        self.predicted_actual = predicted_actual
        self.n_elem = len(predicted_actual)
        self.error = map(lambda (p, a): a - p, self.predicted_actual)
        self.measures = {'MSE': self.mse, 'RMSE': self.rmse, 'MAE': self.mae,
                        'R2': self.r2, 'ExplVar': self.explained_variance}

    def mse(self):
        """
        Se calcula el error cuadrático medio o *Mean Squared Error* (MSE), definido como la suma de las
        diferencias al cuadrado entre el valor actual y el que se predijo para cada uno de los *N* ejemplos,
        tal que:

        :math:`MSE=\\dfrac{1}{N}\displaystyle\sum\limits_{i}^N (p_i - a_i)^2`

        :return: float, que varía entre 0 (mejor) e inf (peor).
        """
        return np.sum(np.square(self.error)) / float(self.n_elem)

    def rmse(self):
        """
        Retorna la raíz cuadrada del valor de MSE, lo cual es útil para independizarse de una escala
        a la hora de comparar el desempeño de distintos modelos.

        :math:`RMSE=\\sqrt{MSE}`

        :return: float, que varía entre 0 (mejor) e inf (peor).
        """
        return np.sqrt(self.mse())

    def mae(self):
        """
        Se calcula el error absoluto medio o *Mean Absolute Error* (MAE), definido como la suma de las
        diferencias absolutas entre el valor actual y el que se predijo para cada uno de los *N* ejemplos,
        tal que:

        :math:`MAE=\\dfrac{1}{N}\displaystyle\sum\limits_{i}^N |p_i - a_i|`

        :return: float, que varía entre 0 (mejor) e inf (peor).
        """
        return np.sum(np.abs(self.error))

    def rmae(self):
        """
        Retorna la raíz cuadrada del valor de MAE, lo cual es útil para independizarse de una escala
        a la hora de comparar el desempeño de distintos modelos.

        :math:`RMAE=\\sqrt{MAE}`

        :return: float, que varía entre 0 (mejor) e inf (peor).
        """

        return np.sqrt(self.mae())

    def r2(self):
        """
        Se calcula el coeficiente de determinación o R^2 el cual indica la proporción de varianza
        de los valores de *actual* que son explicados por las predicciones en *predicted*.

        Ver más info en Wikipedia: `Coefficient of determination
        <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_.

        :return: float, que puede variar entre 0 (peor)  y 1 (mejor).
        """
        mean_actual = np.mean(map(lambda (p, a): a, self.predicted_actual))
        ssres = np.sum(np.square(self.error))
        sstot = np.sum(np.square(map(lambda (p, a): a - mean_actual, self.predicted_actual)))
        return 1 - float(ssres / sstot)

    def explained_variance(self):
        """
        Se calcula la varianza explicada en la predicción sobre los valores reales, tal que:

        :math:`ExpVar=1-\\dfrac{Var(actual - predicted)}{Var(actual)}`

        :return: float, que puede variar entre 0 (peor)  y 1 (mejor).
        """
        var_error = np.var(self.error)
        var_actual = np.var(map(lambda (p, a): a, self.predicted_actual))
        return 1 - float(var_error / var_actual)

    def evaluate(self, measure='R2'):
        """
        Aplica alguna de las medidas implementadas, las cuales se encuentran registradas en el dict
        *self.measures*. Esta función resulta práctica para parametrizar fácilmente la medida a utilizar
        durante el ajuste de un modelo.

        :param measure: string, key de alguna medida implementada.
        :return: float
        """
        assert measure in self.measures.keys(), ValueError('No se encontró la medida '+measure+'.')
        return self.measures[measure]()
