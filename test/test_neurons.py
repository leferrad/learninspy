#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.neurons import LocalNeurons
from learninspy.core.activations import fun_activation, fun_activation_d
from learninspy.core.loss import fun_loss, fun_loss_d
from learninspy.utils.fileio import get_logger
import numpy as np

logger = get_logger(name=__name__)


class TestLocalNeurons(object):
    def __init__(self, shape=(100,100)):
        mat = np.ones(shape)
        self.matrix_neurons = LocalNeurons(mat, shape)

    def test_operators(self):
        logger.info("Testeando operadores...")
        other = np.ones(self.matrix_neurons.shape)
        # Return: LocalNeurons
        res = self.matrix_neurons * other
        assert np.array_equiv(res.matrix, np.ones(self.matrix_neurons.shape))
        res = self.matrix_neurons / other
        assert np.array_equiv(res.matrix, np.ones(self.matrix_neurons.shape))
        res = self.matrix_neurons - other
        assert np.array_equiv(res.matrix, np.zeros(self.matrix_neurons.shape))
        res = self.matrix_neurons + other
        assert np.array_equiv(res.matrix, np.ones(self.matrix_neurons.shape) * 2)
        res = res ** 2  # LocalNeurons
        assert np.array_equiv(res.matrix, np.ones(self.matrix_neurons.shape) * 4)
        assert self.matrix_neurons == LocalNeurons(self.matrix_neurons.matrix, self.matrix_neurons.shape)
        logger.info("OK")

    def test_linalg(self):
        logger.info("Testeando operaciones algebraicas...")
        N = self.matrix_neurons.rows
        other = np.array(range(N))
        # Producto matricial
        res = self.matrix_neurons.mul_array(other)
        sum_range = sum(range(N))
        assert np.array_equiv(res.matrix, np.array([sum_range]*N))
        # Producto punto
        #res = res.transpose().dot(range(N))
        #assert res == (sum_range ** 2) * N
        # Producto entre elementos
        assert self.matrix_neurons.mul_elemwise(self.matrix_neurons) == self.matrix_neurons.collect()
        # Suma entre arreglos
        assert np.array_equiv(self.matrix_neurons.sum_array(self.matrix_neurons), self.matrix_neurons * 2)
        assert self.matrix_neurons.sum() == self.matrix_neurons.count()
        logger.info("OK")

    def test_activation(self):
        logger.info("Testeando la integración de funciones de activación...")
        activation = 'Tanh'
        fun = fun_activation[activation]
        res = self.matrix_neurons.activation(fun)
        fun_d = fun_activation_d[activation]
        res_d = self.matrix_neurons.activation(fun_d)
        assert np.array_equiv(res.matrix, fun(np.ones(self.matrix_neurons.shape)))
        assert np.array_equiv(res_d.matrix, fun_d(np.ones(self.matrix_neurons.shape)))
        # Test de softmax como funcion de activacion
        N = self.matrix_neurons.rows
        shape = (N, 1)
        x = LocalNeurons(range(N), shape)
        res = x.softmax()
        res = map(lambda e: e[0], res.matrix)
        exp_x = np.exp(range(N))
        y = exp_x / float(sum(exp_x))
        assert np.allclose(res, y)
        logger.info("OK")

    def test_loss(self):
        logger.info("Testeando la integración de funciones de activación...")
        loss = 'MSE'
        y = np.array(range(self.matrix_neurons.shape[0]))
        fun = fun_loss[loss]
        res = self.matrix_neurons.loss(fun, y)
        fun_d = fun_loss_d[loss]
        res_d = self.matrix_neurons.loss_d(fun_d, y)
        assert res == fun(np.ones(self.matrix_neurons.shape), y)
        assert np.allclose(res_d.matrix, fun_d(np.ones(self.matrix_neurons.shape), y))
        logger.info("OK")

    # TODO: test regularization stuff (i.e. l1, l2, dropout)

    def test_all(self):
        logger.info("Test de funcionalidades para la clase LocalNeurons")
        self.test_operators()
        self.test_linalg()
        self.test_activation()
        self.test_loss()