#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# For Travis CI compatibility on plots
import matplotlib
matplotlib.use('agg')

from learninspy.core.model import NetworkParameters, NeuralNetwork
from learninspy.core.neurons import LocalNeurons
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import LocalLabeledDataSet, load_ccpp, load_iris
from learninspy.utils.fileio import get_logger
from learninspy.utils.plots import plot_neurons, plot_fitting, plot_activations

import numpy as np

logger = get_logger(name=__name__)

TEMP_PATH = "/tmp/"


class TestNeuralNetwork(object):
    def __init__(self, network_params=None):
        logger.info("Testeo de NeuralNetwork con datos de Combined Cycle Power Plant")
        # Datos
        logger.info("Cargando datos...")
        data = load_ccpp()
        dataset = LocalLabeledDataSet(data)
        self.train, self.valid, self.test = dataset.split_data([.5, .3, .2])
        self.valid = self.valid.collect()

        # Modelo
        if network_params is None:
            network_params = NetworkParameters(units_layers=[4, 30, 1], activation='ReLU',
                                               classification=False, seed=123)
        self.model = NeuralNetwork(network_params)

        # Seteo a mano
        self.model.set_l1(5e-7)
        self.model.set_l2(3e-4)
        self.model.set_dropout_ratios([0.0, 0.0])

    def _fit(self, opt_params=None, stops=None, mini_batch=30, parallelism=2):
        if opt_params is None:
            options = {'step-rate': 1.0, 'decay': 0.995, 'momentum': 0.3, 'offset': 1e-8}
            opt_params = OptimizerParameters(algorithm='Adadelta', options=options)
        if stops is None:
            stops = [criterion['MaxIterations'](30),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        logger.info("Entrenando modelo...")
        hits_valid = self.model.fit(self.train, self.valid, valid_iters=5, mini_batch=mini_batch,
                                    parallelism=parallelism, stops=stops,optimizer_params=opt_params,
                                    keep_best=True, reproducible=True)
        return hits_valid

    def test_check_stops(self):
        criterions = [criterion['MaxIterations'](30),
                      criterion['AchieveTolerance'](0.95, key='hits')]
        self.model.hits_valid = [0.95]  # Hard-coded para simular dichos hits en el fit

        assert self.model.check_stop(epochs=15, criterions=criterions, check_all=False) is True
        assert self.model.check_stop(epochs=15, criterions=criterions, check_all=True) is False
        assert self.model.check_stop(epochs=40, criterions=criterions, check_all=True) is True

    def _test_backprop(self, x, y):
        # Dado que esta funcion se llama en una transformación de RDD, no es rastreada por Python
        # por lo que es mejor hacerle unittest de forma que sea trazada en el coverage code.
        cost, (nabla_w, nabla_b) = self.model._backprop(x, y)
        assert type(cost) is float or isinstance(cost, np.float)
        assert type(nabla_w[0]) is LocalNeurons
        assert type(nabla_b[0]) is LocalNeurons

    def test_fitting(self, opt_params=None, stops=None, mini_batch=30):
        logger.info("Testeando backpropagation en NeuralNetwork...")
        x = self.valid[0].features
        y = [self.valid[0].label]
        self._test_backprop(x, y)

        hits_valid = self._fit(opt_params=opt_params, stops=stops, mini_batch=mini_batch)
        logger.info("Asegurando salidas correctas...")
        assert hits_valid > 0.7

        hits_test, pred_test = self.model.evaluate(self.test, predictions=True, measure='R2')
        assert hits_test > 0.7

        logger.info("OK")

        logger.info("Testeando funciones de prediccion...")
        predicts = self.model.predict(self.test.collect())
        assert np.array_equiv(pred_test, predicts)

        # Test plot of fitting
        logger.info("Testeando plots de fitting...")
        plot_fitting(self.model, show=False)
        logger.info("OK")
        return

    def test_parallelism(self, mini_batch=10):
        logger.info("Testeando variantes del nivel de paralelismo...")

        # Datos
        logger.info("Datos utilizados: Iris")
        data = load_iris()
        dataset = LocalLabeledDataSet(data)
        self.train, self.valid, self.test = dataset.split_data([.5, .3, .2])
        self.valid = self.valid.collect()

        # Optimizacion
        options = {'step-rate': 1.0, 'decay': 0.995, 'momentum': 0.3, 'offset': 1e-8}
        opt_params = OptimizerParameters(algorithm='Adadelta', options=options)
        stops = [criterion['MaxIterations'](10)]

        # Niveles de paralelismo
        parallelism = [-1, 0, 2]

        for p in parallelism:
            logger.info("Seteando paralelismo en %i", p)
            hits_valid = self._fit(opt_params=opt_params, stops=stops, mini_batch=mini_batch, parallelism=p)
            logger.info("Asegurando salidas correctas...")
            assert hits_valid > 0.7

            hits_test, pred_test = self.model.evaluate(self.test, predictions=True, measure='R2')
            assert hits_test > 0.7

            logger.info("OK")
        return


    def test_plotting(self):
        logger.info("Testeando plots de activaciones...")
        plot_activations(self.model.params, show=False)
        logger.info("OK")

        logger.info("Testeando plots de neuronas...")
        plot_neurons(self.model, show=False)
        logger.info("OK")
        return

    def test_fileio_model(self):
        # Save
        model_name = 'test_model.lea'
        self.model.save(filename=TEMP_PATH+model_name)

        # Load
        test_model = NeuralNetwork.load(filename=TEMP_PATH+model_name)

        assert self.model.params == test_model.params