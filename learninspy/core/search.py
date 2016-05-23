#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np

# Dependencias internas
from learninspy.core.activations import fun_activation
from learninspy.core.model import NetworkParameters, NeuralNetwork
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.fileio import get_logger

# Librerias de Python
import os

logger = get_logger(name=__name__)
logger.propagate = False  # Para que no se dupliquen los mensajes por herencia


# optimization_domain = {}  # TODO Soportar esta funcionalidad
network_domain = {'n_layers': ((3, 7), 1),  # Teniendo en cuenta capa de entrada y salida
                  'activation': ['Tanh', 'ReLU', 'Softplus'],  # Tambien puede ponerse fun_activation.keys()
                  'dropout_ratios': ((0.0, 0.7), 1),  # ((begin, end), precision)
                  'l1': ((1e-6, 1e-4), 6), 'l2': ((1e-6, 1e-3), 4),  # ((begin, end), precision)
                  'perc_neurons': ((0.2, 1.2), 2)
                  }


class RandomSearch(object):
    def __init__(self, net_params, n_layers=0, n_iter=10, net_domain=None, seed=123):
        self.net_params = net_params
        if net_domain is None:
            net_domain = network_domain
        self.domain = net_domain
        self.n_iter = n_iter
        self.n_layers = n_layers
        self.rng = np.random.RandomState(seed)
        self.seeds = list(self.rng.randint(0, 1000, size=n_iter))

    def _sample_units_layers(self):
        if self.n_layers == 0:  # Debe quedar tal cual esta la config de capas neuronales
            units_layers = self.net_params.units_layers
        else:
            dom_layers = self.domain['n_layers']
            dom_neurons = self.domain['perc_neurons']
            if self.n_layers == -1:  # Se elige en random la cant de capas
                n_layers = self.rng.randint(low=dom_layers[0][0], high=dom_layers[0][1])
            else:
                n_layers = self.n_layers
            """
            Lo siguiente es asi: necesito generar una lista de porcentajes, cuyo intervalo se da por la tupla
            dom_neurons[0] y la precision del float por dom_neurons[1]. Dichos porcentajes se aplican desde la capa
            de entrada para dar una idea de expansion o compresion de la cant de neuronas con respecto a la anterior.
            Por ej: Se tienen 500 neuronas de entrada y 3 de salida, y se quieren 5 capas. Entonces con la lista
            de porcentajes generada para un dom_neurons=((0.1,1.5),1):
            [0.5,0.2,1.0,1.2] (notando que se cubre el rango [0.1, 1.5] con precision de 1 decimal)
            se va a obtener la siguiente lista de neuronas por capa:
            [500, 250, 50, 50, 60, 3]
            """
            percents = map(lambda p: round(p, dom_neurons[1]),
                           self.rng.uniform(low=dom_neurons[0][0], high=dom_neurons[0][1], size=n_layers-2))
            n_in = self.net_params.units_layers[0]
            n_out = self.net_params.units_layers[-1]
            units_layers = [n_in] # Primero va la capa de entrada
            for p in percents:
                # Agrego unidades ocultas, siendo un porcentaje de la capa anterior
                units_layers.append(int(p * units_layers[-1]))
            units_layers.append(n_out)  # Por ultimo la capa de salida
            # TODO: verificar que haya una lista válida de neuronas
        return units_layers

    def _sample_activation(self, n_layers):
        act = self.net_params.activation[-1]  # Tomo una activacion de referencia
        if type(act) is bool:  # Debe elegirse en modo random
            dom_activations = self.domain['activation']
            if act is True:  # Quiere decir que todas las activaciones deben ser iguales
                index = self.rng.randint(low=0, high=len(dom_activations))
                sample_act = dom_activations[index]
                activation = [sample_act] * (n_layers - 1)
            else:  # Las activaciones pueden ser distintas entre capas
                index = self.rng.randint(low=0, high=len(dom_activations), size=n_layers-1)
                activation = [dom_activations[i] for i in index]
        else:  # Ya vienen definidas las activaciones, las dejo como estan
            activation = self.net_params.activation
        return activation

    def _sample_dropout_ratios(self, n_layers):
        if type(self.net_params.dropout_ratios) is bool:  # Se debe muestrar
            dom_dropout = self.domain['dropout_ratios']
            range_dropout = dom_dropout[0]
            precision = dom_dropout[1]
            dropout_ratios = self.rng.uniform(low=range_dropout[0], high=range_dropout[1], size=n_layers-1)
            dropout_ratios = map(lambda d: round(d, precision), dropout_ratios)
            if self.net_params.classification is True:
                dropout_ratios[-1] = 0.0  # Ya que no debe haber dropout para Softmax
        else:
            dropout_ratios = self.net_params.dropout_ratios
            if len(dropout_ratios) != (n_layers - 1):  # Longitud distinta a la requerida
                dropout_ratios = [0.2] + [0.5] * (n_layers-3) + [0.0]  # Dejo esta config por defecto (chau la otra)
        return dropout_ratios

    def _sample_l1_l2(self):
        # L1
        if type(self.net_params.strength_l1) is bool:
            dom_l1 = self.domain['l1']
            range_l1 = dom_l1[0]
            precision = dom_l1[1]
            strength_l1 = self.rng.uniform(low=range_l1[0], high=range_l1[1])
            strength_l1 = round(strength_l1, precision)
        else:
            strength_l1 = self.net_params.strength_l1

        # L2
        if type(self.net_params.strength_l2) is bool:
            dom_l2 = self.domain['l2']
            range_l2 = dom_l2[0]
            precision = dom_l2[1]
            strength_l2 = self.rng.uniform(low=range_l2[0], high=range_l2[1])
            strength_l2 = round(strength_l2, precision)
        else:
            strength_l2 = self.net_params.strength_l2
        return strength_l1, strength_l2

    def _take_sample(self, seed=123):
        self.rng.seed(seed)  # Alimento generador random con semilla
        units_layers = self._sample_units_layers()
        activation = self._sample_activation(n_layers=len(units_layers))
        dropout_ratios = self._sample_dropout_ratios(n_layers=len(units_layers))
        strength_l1, strength_l2 = self._sample_l1_l2()
        net_params = NetworkParameters(units_layers=units_layers, activation=activation,
                                       dropout_ratios=dropout_ratios, classification=self.net_params.classification,
                                       strength_l1=strength_l1, strength_l2=strength_l2, seed=seed)
        return net_params

    def fit(self, train, valid, test, mini_batch=100, parallelism=4, stops=None, optimizer_params=None, keep_best=True):
        """


        .. note:: Los parámetros son los mismos que recibe la función :func:`~learninspy.core.model.NeuralNetwork.fit`
            incluyendo también el conjunto de prueba *test* que se utiiza para validar la conveniencia de
            cada modelo logrado.
        """
        if stops is None:
            stops = [criterion['MaxIterations'](10),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        if optimizer_params is None:
            local_stops = [criterion['MaxIterations'](10),
                           criterion['AchieveTolerance'](0.90, key='hits')]
            optimizer_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, merge_criter='w_avg')

        # Para comparar y quedarse el mejor modelo
        best_model = None
        best_hits = 0.0

        logger.info("Optimizacion utilizada: %s", str(optimizer_params))
        for it in xrange(self.n_iter):
            net_params_sample = self._take_sample(seed=self.seeds[it])
            logger.info("Iteracion %i en busqueda.", it+1)
            logger.info("Configuracion usada: %s", os.linesep+str(net_params_sample))
            neural_net = NeuralNetwork(net_params_sample)
            hits_valid = neural_net.fit(train, valid, mini_batch=mini_batch, parallelism=parallelism,
                                        stops=stops, optimizer_params=optimizer_params, keep_best=keep_best)
            hits_test = neural_net.evaluate(test, predictions=False)
            if hits_test >= best_hits:
                best_hits = hits_test
                best_model = neural_net
        neural_net = best_model
        hits_test = best_hits
        return neural_net, hits_test

    @staticmethod
    def save_config():
        pass


def _test_busqueda():
    net_params = NetworkParameters(units_layers=[100,5,3], activation=False, dropout_ratios=True, classification=True,
                                   strength_l1=True, strength_l2=True)
    rndsearch = RandomSearch(net_params, n_layers=0, n_iter=40)
    rndsearch.fit(None, None, None)
    return

#_test_busqueda()