#!/usr/bin/env python
# -*- coding: utf-8 -*-7

"""
Este módulo contiene funcionalidades para monitorear el corte del entrenamiento
en una red neuronal. El mismo está importantemente basado en el excelente
package de optimización `climin <https://github.com/BRML/climin>`_.

En cualquier aplicación de aprendizaje maquinal, por lo general no se ejecuta
la optimización de un modelo hasta obtener un desempeño deseado ya que puede
ser que no se alcance dicho objetivo por la configuración establecida. Es por
ello que resulta conveniente establecer ciertas heurísticas para monitorear la
convergencia del modelo en su optimización.

Un criterio de corte es una función que utiliza información de la optimización
de un modelo durante dicho proceso (e.g. scoring sobre el conjunto de validación,
costo actual, cantidad de iteraciones realizadas) y, en base a una regla establecida,
determina si se debe frenar o no dicho proceso. En su implementación, son instanciados
al crearse con los parámetros que determinen su configuración y luego se utilizan
llamándolos con un parámetro que es un dict con los key, values necesarios.

Una característica interesante de estos criterios es que se pueden combinar con
operaciones lógicas de and/or, de forma que el corte de la optimización se realice
cuando todos los criterios elegidos, o alguno de ellos, lo determinen. Además se
ofrece la facilidad de utilizar estos criterios utilizando el diccionario
*criterion* instanciado en este módulo, con lo cual se combinan sencillamente
llamándolos mediante strings (es así como se utilizan en los módulos de learninspy).

>>> from learninspy.core.stops import criterion
>>> criterions = [criterion['MaxIterations'](10),\ ...
>>>               criterion['AchieveTolerance'](0.9, 'hits'),\ ...
>>>               criterion['NotBetterThanAfter'](0.6, 5, 'hits')]
>>> results = {'hits': 0.8, 'iterations': 8}
>>> stop = all(map(lambda c: c(results), criterions))
>>> print stop
False
>>> results = {'hits': 15, 'iterations': 0.95}
>>> stop = any(map(lambda c: c(results), criterions))
>>> print stop
True
"""

import signal
import time


class MaxIterations(object):
    """
    Criterio para frenar la optimización luego de un máximo de iteraciones definido.

    :param max_iter: int, máximo de iteraciones.
    """

    def __init__(self, max_iter):
        self.max_iter = max_iter

    def __call__(self, results):
        return results['iterations'] >= self.max_iter

    def __str__(self):
        return "Stop at a maximum of "+str(self.max_iter)+" iterations."


class AchieveTolerance(object):
    """
    Criterio para frenar la optimización luego de alcanzar un valor de tolerancia
    sobre una cierta variable de la optimización.

    :param tolerance: float, tolerancia en rango *0 < tolerance <= 1*.
    :param key: string, correspondiente a donde se aplica la tolerancia ('cost' o  'hits').

    >>> # Tolerancia sobre el costo de optimización
    >>> tol = 1e-3
    >>> stop = AchieveTolerance(tol, key='cost')
    >>> ...
    >>> # Tolerancia sobre el resultado parcial de evaluación
    >>> tol = 0.9
    >>> stop = AchieveTolerance(tol, key='hits')
    """

    def __init__(self, tolerance, key='hits'):
        self.tolerance = tolerance
        self.key = key

    def __call__(self, results):
        achieved = results[self.key] >= self.tolerance  # self.key == 'hits':
        if self.key == 'cost':
            achieved = results[self.key] <= self.tolerance
        return achieved

    def __str__(self):
        return "Stop when a tolerance of "+str(self.tolerance)+" is achieved/exceeded in "+self.key+"."


class ModuloNIterations(object):
    """
    Criterio para frenar la optimización cuando una iteración alcanzada
    es módulo del parámetro *n* ingresado. Este criterio es útil al
    combinarse con otros más.

    :param n: int.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, results):
        return results['iterations'] % self.n == 0

    def __str__(self):
        return "Stop when an iteration is modulo of "+str(self.n)+"."


class TimeElapsed(object):
    """
    Criterio para frenar la optimización luego de superar un lapso de tiempo fijado.

    :param sec: float, lapso de tiempo máximo, en unidad de **segundos**.
    """

    def __init__(self, sec):
        self.sec = sec
        self.start = time.time()

    def __call__(self, results=None):
        return time.time() - self.start > self.sec

    def __str__(self):
        return "Stop when "+str(self.sec)+" seconds have elapsed."


class NotBetterThanAfter(object):
    """
    Criterio para frenar la optimización cuando luego de una cierta cantidad de
    iteraciones no se alcanzó un mínimo valor determinado sobre una variable
    de la optimización.

    :param minimal: float, valor mínimo a alcanzar o superar sobre la variable definida por *key*.
    :param after: int, cantidad de iteraciones a partir de las cuales medir sobre *minimal*.
    :param key: string, correspondiente a donde se aplica la diferencia con *minimal* ('cost' o  'hits').
    """

    def __init__(self, minimal, after, key='hits'):
        self.minimal = minimal
        self.after = after
        self.key = key

    def __call__(self, results):
        not_better = results[self.key] < self.minimal
        if self.key == 'cost':
            not_better = results[self.key] > self.minimal
        return results['iterations'] > self.after and not_better is True

    def __str__(self):
        return "Stop when "+self.key+" does not improve a minimal of " + \
               str(self.minimal)+" after "+str(self.after)+" iterations."

# TODO: Hacer un NoImprovementAfter que cada N iteraciones tome un máximo y un mínimo y compruebe que se mejoró en X porciento
# TODO: Hacer un ContinueFromBest que luego de N iteraciones siga la optimizacion con el best model logrado.


class Patience(object):
    """
    Criterio para frenar la optimización siguiendo el método heurístico
    de paciencia ideado por Bengio [bengio2012practical]_.

    Se basa en incrementar el número de iteraciones en la optimización
    multiplicando por un factor *grow_factor* y/o sumando una constante
    *grow_offset* una vez que se obtiene un mejor valor de la variable
    a optimizar dada por *key*. Esto es realizado para que heurísticamente
    se le tenga paciencia en la optimización a un nuevo candidato encontrado
    incrementando su tiempo de ajuste.
    ...

    :param initial: int, número de iteración a partir de la cual medir la "paciencia".
    :param key: string, correspondiente a donde se mide el progreso ('cost' o  'hits').
    :param grow_factor: float, debe ser distinto de 1 si grow_offset == 0.
    :param grow_offset: float, debe ser distinto de 0 si grow_factor == 1.
    :param threshold: float, umbral de diferencia entre el valor actual y el mejor obtenido.

    **Referencias**:

    .. [bengio2012practical] Bengio, Y. (2012). Practical recommendations for gradient-based training of deep architectures.
                             In Neural Networks: Tricks of the Trade (pp. 437-478). Springer Berlin Heidelberg.
    """
    def __init__(self, initial, key='hits', grow_factor=1, grow_offset=0,
                 threshold=0.05):
        if grow_factor == 1 and grow_offset == 0:
            raise ValueError('Se necesita especificar o bien un grow_factor != 1'
                             'o un grow_offset != 0')
        self.key = key
        self.patience = initial
        self.grow_factor = grow_factor
        self.grow_offset = grow_offset
        self.threshold = threshold

        self.best_value = float('inf')
        if self.key == 'hits':
            self.best_value = -self.best_value  # Se busca maximizar key

    def __call__(self, results):
        i = results['iterations']
        value = results[self.key]
        if self.key == 'hits':
            # Se busca maximizar key
            better_value = value > self.best_value

        else:
            # Se busca minimizar key (que es 'cost')
            better_value = value < self.best_value
        if better_value is True:
            difference = (value - self.best_value) > self.threshold
            if self.key == 'cost':
                difference = (value - self.best_value) < self.threshold

            if difference is True and i > 0:
                self.patience = max(i * self.grow_factor + self.grow_offset,
                                    self.patience)
            self.best_value = value
        return i >= self.patience


class OnSignal(object):
    """
    Criterio para frenar la optimización cuando se ejecuta un comando
    que accione una señal del SO (e.g. Ctrl+C para interrupción).

    Útil para detener la optimización a demanda sin perder
    el progreso logrado.

    :param sig: signal, opcional [default: signal.SIGINT].
    """

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig
        self.stopped = False
        self._register()

    def _register(self):
        self.prev_handler = signal.signal(self.sig, self._handler)

    def _handler(self, sig, frame):
        self.stopped = True

    def __call__(self, results=None):
        res, self.stopped = self.stopped, False
        return res

    def __del__(self):
        signal.signal(self.sig, self.prev_handler)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._register()


criterion = {'MaxIterations': MaxIterations, 'AchieveTolerance': AchieveTolerance,
             'ModuloNIterations': ModuloNIterations, 'TimeElapsed': TimeElapsed,
             'NotBetterThanAfter': NotBetterThanAfter, 'Patience': Patience,
             'OnSignal': OnSignal}
