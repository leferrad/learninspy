#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np

# Dependencias internas
import learninspy.core.activations as act
import learninspy.core.loss as loss


class CheckGradientActivation(object):
    """
    Clase para chequear la correcta implementación del gradiente de una función de activación perteneciente a
    :mod:`~learninspy.core.activations`.

    Clase para chequear la correcta implementación del gradiente de una función, mediante diferenciación numérica.
    Basado en http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization

    :param function: string, correspondiente a la key de un dict de funciones.

    >>> activation = 'Tanh'
    >>> check_activation = CheckGradientActivation(activation)
    >>> good_gradient = check_activation()
    >>> assert good_gradient is True, AssertionError("Gradiente de activación "+activation+" mal implementado!")
    """
    def __init__(self, function):
        self.fun = act.fun_activation[function]
        self.fun_d = act.fun_activation_d[function]

    def __call__(self, *args, **kwargs):
        return self.check_numerical_gradient()

    @classmethod
    def _compute_numerical_gradient(cls, J, theta, **kwargs):
        epsilon = 1e-4
        num_grad = map(lambda o: (J(o + epsilon) - J(o - epsilon)) / (2.0 * epsilon), theta)  # Diferencias centradas
        num_grad = np.array(num_grad)
        return num_grad

    def check_numerical_gradient(self):
        theta = np.array([0.5, 0.0001, 1.5, 2.0, -1.0])  # Ejemplo para probar gradientes
        analytic_grad = np.array(map(lambda o: self.fun_d(o), theta))
        numerical_grad = self._compute_numerical_gradient(self.fun, theta)
        diff = np.linalg.norm(numerical_grad - analytic_grad) / np.linalg.norm(numerical_grad + analytic_grad)
        return diff <= 1e-8


class CheckGradientLoss(object):
    """
    .. note:: Experimental, sin terminar.
    """
    def __init__(self, function):
        self.fun = loss.fun_loss[function]
        self.fun_d = loss.fun_loss_d[function]

    def __call__(self, *args, **kwargs):
        return self.check_numerical_gradient()

    @classmethod
    def _compute_numerical_gradient(cls, J, theta, target):
        epsilon = 1e-4
        n = len(theta)
        numgrad = np.zeros(n)
        for i in xrange(n):
            offset = np.zeros(n)
            offset[i] = epsilon
            offset_pos = theta + offset
            offset_neg = theta - offset
            numgrad[i] = (J(offset_pos, target) - J(offset_neg, target)) / (2.0 * epsilon)
        return numgrad

    def check_numerical_gradient(self):
        theta = np.array([0.5, 0.0001, 0.2, 1.0, 0.1])  # Ejemplo para probar gradientes
        if self.fun == loss.fun_loss['CrossEntropy']:  # Si el error es la entropia cruzada modifico el ejemplo
            theta = np.exp(theta) / float(sum(np.exp(theta)))  # Porque la derivada es con respecto a la salida del softmax
        target = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # Ejemplo de target
        analytic_grad = self.fun_d(theta, target)
        numerical_grad = self._compute_numerical_gradient(self.fun, theta, target)
        diff = np.linalg.norm(numerical_grad - analytic_grad) / np.linalg.norm(numerical_grad + analytic_grad)
        bad_grad = diff > 1e-8 or np.isnan(diff) or np.isinf(diff)
        return not bad_grad
