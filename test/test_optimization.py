#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.model import NetworkParameters, NeuralNetwork
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import LocalLabeledDataSet, load_iris
from learninspy.utils.fileio import get_logger

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

    def _optimize(self, stops=None, mini_batch=30, parallelism=2):
        if stops is None:
            stops = [criterion['MaxIterations'](30),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        logger.info("Entrenando modelo...")
        hits_valid = self.model.fit(self.train, self.valid, valid_iters=1, mini_batch=mini_batch,
                                    parallelism=parallelism, stops=stops,optimizer_params=self.opt_params,
                                    keep_best=True, reproducible=True)
        return hits_valid

    def test_optimization(self, stops=None, mini_batch=30, parallelism=1):
        hits_valid = self._optimize(stops=stops, mini_batch=mini_batch, parallelism=parallelism)
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