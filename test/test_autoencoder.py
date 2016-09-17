#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# For Travis CI compatibility on plots
import matplotlib
matplotlib.use('agg')

from learninspy.core.model import NetworkParameters
from learninspy.core.neurons import LocalNeurons
from learninspy.core.autoencoder import AutoEncoder, StackedAutoencoder
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import LocalLabeledDataSet
from learninspy.utils.fileio import get_logger
from learninspy.utils.plots import plot_neurons, plot_fitting, plot_activations

import numpy as np

import os

logger = get_logger(name=__name__)


class TestAutoEncoder(object):
    def __init__(self, network_params=None, dropout_in=0.0):
        logger.info("Testeo de AutoEncoder con datos de Iris")
        # Datos
        logger.info("Cargando datos...")
        dataset = LocalLabeledDataSet()
        path = os.path.abspath(os.path.join(os.path.realpath(__file__),
                                            os.path.pardir,
                                            os.pardir,
                                            'examples/datasets/iris.csv'))
        dataset.load_file(path)
        self.train, self.valid, self.test = dataset.split_data([.5, .3, .2])
        self.train = self.train.collect()
        self.valid = self.valid.collect()

        # Modelo
        if network_params is None:
            network_params = NetworkParameters(units_layers=[4, 8], activation=['ReLU', 'ReLU'],
                                               classification=False)
        self.model = AutoEncoder(network_params, dropout_in=dropout_in)

    def _fit(self, opt_params=None, stops=None, mini_batch=30, parallelism=2):
        if opt_params is None:
            opt_params = OptimizerParameters(algorithm='Adadelta')
        if stops is None:
            stops = [criterion['MaxIterations'](30),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        logger.info("Entrenando modelo...")
        hits_valid = self.model.fit(self.train, self.valid, valid_iters=1, mini_batch=mini_batch,
                                    parallelism=parallelism, stops=stops,optimizer_params=opt_params,
                                    keep_best=True, reproducible=True)
        return hits_valid

    def _test_backprop(self, x, y):
        # Dado que esta funcion se llama en una transformación de RDD, no es rastreada por Python
        # por lo que es mejor hacerle unittest de forma que sea trazada en el coverage code.
        cost, (nabla_w, nabla_b) = self.model._backprop(x, y)
        assert type(cost) is float or isinstance(cost, np.float)
        assert type(nabla_w[0]) is LocalNeurons
        assert type(nabla_b[0]) is LocalNeurons

    def test_fitting(self, opt_params=None, stops=None, mini_batch=30, parallelism=1):
        logger.info("Testeando backpropagation en AutoEncoders...")
        x = self.train[0].features
        y = self.train[0].label
        self._test_backprop(x, y)

        hits_valid = self._fit(opt_params=opt_params, stops=stops, mini_batch=mini_batch, parallelism=parallelism)
        logger.info("Asegurando salidas correctas...")
        assert hits_valid > 0.8
        hits_test = self.model.evaluate(self.test, predictions=False, measure='R2')
        assert hits_test > 0.8
        logger.info("OK")


class TestStackedAutoEncoder(object):
    def __init__(self, network_params=None, dropout=None):
        logger.info("Testeo de StackedAutoEncoder con datos de Iris")
        # Datos
        logger.info("Cargando datos...")
        dataset = LocalLabeledDataSet()
        path = os.path.abspath(os.path.join(os.path.realpath(__file__),
                                            os.path.pardir,
                                            os.pardir,
                                            'examples/datasets/iris.csv'))
        dataset.load_file(path)
        self.train, self.valid, self.test = dataset.split_data([.5, .3, .2])
        self.train = self.train.collect()
        self.valid = self.valid.collect()

        # Modelo
        if network_params is None:
            network_params = NetworkParameters(units_layers=[4, 10, 3], activation=['ReLU', 'ReLU'],
                                               strength_l1=1e-5, strength_l2=3e-4,
                                               dropout_ratios=[0.0, 0.0], classification=True)
        self.model = StackedAutoencoder(network_params, dropout=dropout)

    def _fit(self, opt_params=None, stops=None, mini_batch=30, parallelism=2):
        if opt_params is None:
            options = {'step-rate': 1.0, 'decay': 0.9, 'momentum': 0.0, 'offset': 1e-8}
            opt_params = OptimizerParameters(algorithm='Adadelta', options=options)
        if stops is None:
            stops = [criterion['MaxIterations'](30),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        logger.info("Entrenando modelo...")
        hits_valid = self.model.fit(self.train, self.valid, valid_iters=1, mini_batch=mini_batch,
                                    parallelism=parallelism, stops=stops,optimizer_params=opt_params,
                                    keep_best=True, reproducible=True)
        return hits_valid

    def _finetune(self, opt_params=None, stops=None, mini_batch=30, parallelism=2):
        if opt_params is None:
            options  = {'step-rate': 1.0, 'decay': 0.9, 'momentum': 0.0, 'offset': 1e-8}
            opt_params = OptimizerParameters(algorithm='Adadelta', options=options)
        if stops is None:
            stops = [criterion['MaxIterations'](30),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        logger.info("Entrenando modelo...")
        hits_valid = self.model.finetune(self.train, self.valid, valid_iters=1, mini_batch=mini_batch,
                                         parallelism=parallelism, stops=stops,optimizer_params=opt_params,
                                         keep_best=True, reproducible=True)
        return hits_valid

    def test_fitting(self, opt_params=None, stops=None, mini_batch=30, parallelism=1):
        hits_valid_pretrain = self._fit(opt_params=opt_params, stops=stops,
                                        mini_batch=mini_batch, parallelism=parallelism)
        hits_test_pretrain = self.model.evaluate(self.test, predictions=False, measure='F-measure')
        hits_valid_finetune = self._finetune(opt_params=opt_params, stops=stops,
                                             mini_batch=mini_batch, parallelism=parallelism)
        hits_test_finetune = self.model.evaluate(self.test, predictions=False, measure='F-measure')
        logger.info("Asegurando salidas correctas...")
        assert hits_valid_pretrain > 0.6
        assert hits_valid_finetune > 0.8
        assert hits_test_pretrain > 0.6
        assert hits_test_finetune > 0.8
        logger.info("OK")

        # Test plot of fitting
        logger.info("Testeando plots de fitting...")
        plot_fitting(self.model, show=False)
        logger.info("OK")
        return

    def test_plotting(self):
        logger.info("Testando plots de activaciones...")
        plot_activations(self.model.params, show=False)
        logger.info("OK")

        logger.info("Testeando plots de neuronas...")
        plot_neurons(self.model, show=False)
        logger.info("OK")
        return