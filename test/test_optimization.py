#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.model import NetworkParameters, NeuralNetwork
from learninspy.core.optimization import OptimizerParameters, optimize, mix_models
from learninspy.core.stops import criterion
from learninspy.utils.data import LocalLabeledDataSet, load_iris
from learninspy.utils.fileio import get_logger

import copy

logger = get_logger(name=__name__)


class TestOptimizer(object):
    def __init__(self, opt_params=None):
        logger.info("Testeo de Optimizer con datos de Iris")
        # Datos
        logger.info("Cargando datos...")
        data = load_iris()
        dataset = LocalLabeledDataSet(data)
        self.train, self.valid, self.test = dataset.split_data([.5, .3, .2])
        self.train = self.train.collect()
        self.valid = self.valid.collect()
        self.test = self.test.collect()

        # Configuracion de optimizacion
        if opt_params is None:  # Por defecto, se utiliza Adadelta
            stops = [criterion['MaxIterations'](10),
                     criterion['AchieveTolerance'](0.95, key='hits')]
            options = {'step-rate': 1.0, 'decay': 0.99, 'momentum': 0.3, 'offset': 1e-8}
            opt_params = OptimizerParameters(algorithm='Adadelta', stops=stops,
                                             options=options, merge_criter='w_avg')
        self.opt_params = opt_params

        # Configuracion de modelo
        net_params = NetworkParameters(units_layers=[4, 10, 3], activation='ReLU', strength_l1=1e-5, strength_l2=3e-4,
                                       dropout_ratios=[0.2, 0.0], classification=True)
        self.model = NeuralNetwork(net_params)

    def _optimize(self, stops=None, mini_batch=30, parallelism=1, keep_best=True):
        if stops is None:
            stops = [criterion['MaxIterations'](50),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        logger.info("Entrenando modelo...")
        if keep_best is True:
            best_valid = 0.0
            best_model = None
        epoch = 0
        hits_valid = 0.0
        while self.model.check_stop(epoch, stops) is False:
            hits_train = optimize(self.model, self.train, params=self.opt_params, mini_batch=mini_batch, seed=123)
            hits_valid = self.model.evaluate(self.valid, predictions=False, measure='F-measure')
            if keep_best is True:
                if hits_valid >= best_valid:
                    best_valid = hits_valid
                    best_model = self.model
            epoch += 1
        if keep_best is True:
            if best_model is not None: # Si hubo algun best model, procedo con el reemplazo
                self.model = copy.deepcopy(best_model)

        return hits_valid

    def test_optimization(self, stops=None, mini_batch=30, parallelism=1, keep_best=True):
        hits_valid = self._optimize(stops=stops, mini_batch=mini_batch, parallelism=parallelism, keep_best=keep_best)
        logger.info("Asegurando salidas correctas...")
        assert hits_valid > 0.8
        hits_test = self.model.evaluate(self.test, predictions=False, measure='F-measure')
        assert hits_test > 0.8
        logger.info("OK")


class TestAdadelta(TestOptimizer):
    def __init__(self, opt_params=None):
        if opt_params is None:
            stops = [criterion['MaxIterations'](10),
                     criterion['AchieveTolerance'](0.95, key='hits')]
            options = {'step-rate': 1.0, 'decay': 0.99, 'momentum': 0.3, 'offset': 1e-8}
            opt_params = OptimizerParameters(algorithm='Adadelta', stops=stops,
                                             options=options, merge_criter='w_avg')
        super(TestAdadelta, self).__init__(opt_params)


class TestGDStandard(TestOptimizer):
    def __init__(self, opt_params=None):
        if opt_params is None:
            stops = [criterion['MaxIterations'](10),
                     criterion['AchieveTolerance'](0.95, key='hits')]
            options = {'step-rate': 0.01, 'momentum': 0.8, 'momentum_type': 'standard'}
            opt_params = OptimizerParameters(algorithm='GD', stops=stops,
                                             options=options, merge_criter='w_avg')
        super(TestGDStandard, self).__init__(opt_params)


class TestGDNesterov(TestOptimizer):
    def __init__(self, opt_params=None):
        if opt_params is None:
            stops = [criterion['MaxIterations'](10),
                     criterion['AchieveTolerance'](0.95, key='hits')]
            options = {'step-rate': 0.01, 'momentum': 0.8, 'momentum_type': 'nesterov'}
            opt_params = OptimizerParameters(algorithm='GD', stops=stops,
                                             options=options, merge_criter='w_avg')
        super(TestGDNesterov, self).__init__(opt_params)


def test_mix_models():
    # Dado que esta funcion se llama en una transformación de RDD, no es rastreada por Python
    # por lo que es mejor hacerle unittest de forma que sea trazada en el coverage code.

    # Configuracion de modelo
    net_params = NetworkParameters(units_layers=[4, 10, 3], activation='ReLU', strength_l1=1e-5, strength_l2=3e-4,
                                   dropout_ratios=[0.2, 0.0], classification=True)
    model = NeuralNetwork(net_params)
    mixed_layers = mix_models(model.list_layers, model.list_layers)
    for l in xrange(len(mixed_layers)):
        assert mixed_layers[l] == model.list_layers[l] * 2  # A + A = 2A