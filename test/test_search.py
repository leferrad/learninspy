#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.model import NetworkParameters, NeuralNetwork
from learninspy.core.optimization import OptimizerParameters, optimize
from learninspy.core.stops import criterion
from learninspy.core.search import RandomSearch
from learninspy.utils.data import LocalLabeledDataSet, load_iris
from learninspy.utils.fileio import get_logger

logger = get_logger(name=__name__)


class TestRandomSearch(object):
    def __init__(self, type_model=None, net_params=None, opt_params=None, n_layers=0, n_iter=10, seed=123):

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

        # Configuracion de modelo a optimizar
        if type_model is None:
            type_model = NeuralNetwork
        self.type_model = type_model
        if net_params is None:
            net_params = NetworkParameters(units_layers=[4, 10, 3], activation=False, dropout_ratios=True,
                                           classification=True, strength_l1=True, strength_l2=True, seed=seed)
        self.net_params = net_params

        # Configuracion del Random Search
        self.rnd_search = RandomSearch(self.net_params, n_layers, n_iter, net_domain=None, seed=seed)

    def test_good_fitting(self, mini_batch=30, parallelism=1):
        best_model, hits_test = self.rnd_search.fit(self.type_model, self.train, self.valid, self.test,
                                                    mini_batch=mini_batch, parallelism=parallelism, valid_iters=1,
                                                    stops=None, optimizer_params=self.opt_params, reproducible=True,
                                                    keep_best=True)
        logger.info("Asegurando salidas correctas...")
        assert hits_test > 0.8
        logger.info("OK")

    def test_all_fitting(self, mini_batch=30, parallelism=1):
        n_layers = [-1, 0, 3, 4]
        best_hits_final = 0.0
        self.rnd_search.n_iter = 1
        for n_l in n_layers:
            logger.info("Testeando con 'n_layers = %i'", n_l)
            self.rnd_search.n_layers = n_l
            best_model, best_hits = self.rnd_search.fit(self.type_model, self.train, self.valid, self.test,
                                                        mini_batch=mini_batch, parallelism=parallelism, valid_iters=1,
                                                        stops=None, optimizer_params=self.opt_params, reproducible=True,
                                                        keep_best=True)
            if best_hits > best_hits_final:
                best_hits_final = best_hits

        logger.info("Asegurando salidas correctas...")
        assert best_hits_final > 0.3
        logger.info("OK")