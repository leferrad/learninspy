from dnn import loss

__author__ = 'leferrad'

import numpy as np

import dnn.activations as act
import dnn.loss as loss


class CheckGradient(object):
    def __init__(self, functions):
        n_fun = len(functions)
        self.fun = [None] * n_fun
        self.fun_d = [None] * n_fun

    def __call__(self, *args, **kwargs):
        bad_grads = self.check_numericalgradient()
        if not any(bad_grads):  # Si no hay ningun gradiente mal implementado
            bad_grads = None  # Retorno que ningun gradiente esta mal implementado
        return bad_grads

    # Ver http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
    def compute_numericalgradient(self, J, theta):
        epsilon = 1e-4
        numgrad = map(lambda o: (J(o + epsilon) - J(o - epsilon)) / (2.0 * epsilon), theta)  # Diferencias centradas
        numgrad = np.array(numgrad)
        return numgrad

    def check_numericalgradient(self):
        theta = np.array([0.5, 0.0001, 1.5, 2.0, -1.0])  # Ejemplo para probar gradientes
        n_fun = len(self.fun)
        bad_grads = np.zeros(n_fun,dtype=bool)
        for i in xrange(n_fun):  # Chequeo la activacion de cada capa
            analytic_grad = np.array(map(lambda o: self.fun_d[i](o), theta))
            numerical_grad = self.compute_numericalgradient(self.fun[i], theta)
            diff = np.linalg.norm(numerical_grad - analytic_grad) / np.linalg.norm(numerical_grad + analytic_grad)
            bad_grads[i] = diff > 1e-8
        return bad_grads


class CheckGradientActivation(CheckGradient):
    def __init__(self, functions):
        CheckGradient.__init__(self, functions)
        for i in xrange(len(functions)):
            self.fun[i] = act.fun_activation[functions[i]]
            self.fun_d[i] = act.fun_activation_d[functions[i]]


class CheckGradientLoss(CheckGradient):
    def __init__(self, function):
        CheckGradient.__init__(self, function)
        self.fun = loss.fun_loss[function]
        self.fun_d = loss.fun_loss_d[function]

    def __call__(self, *args, **kwargs):
        bad_grad = self.check_numericalgradient()
        if not bad_grad:  # Si no hay gradiente mal implementado
            bad_grad = None  # Retorno que ningun gradiente esta mal implementado
        return bad_grad

    # Ver http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
    def compute_numericalgradient(self, J, theta, target):
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

    def check_numericalgradient(self):
        theta = np.array([0.5, 0.0001, 0.2, 1.0, 0.1])  # Ejemplo para probar gradientes
        if self.fun == loss.fun_loss['CrossEntropy']:  # Si el error es la entropia cruzada modifico el ejemplo
            theta = act.softmax(theta)  # Porque la derivada es con respecto a la salida del softmax
        target = np.array([0.0, 0.0, 0.0, 1.0, 0.0]) # Ejemplo de target
        analytic_grad = self.fun_d(theta, target)
        numerical_grad = self.compute_numericalgradient(self.fun, theta, target)
        diff = np.linalg.norm(numerical_grad - analytic_grad) / np.linalg.norm(numerical_grad + analytic_grad)
        bad_grad = diff > 1e-8 or np.isnan(diff) or np.isinf(diff)
        return bad_grad
