#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
En este módulo se implementan los mecanismos para realizar búsquedas
de los parámetros utilizados durante el ajuste de redes neuronales.

Como puede notarse durante el diseño del modelo en cuestión, existen diversos
parámetros y configuraciones que se deben especificar en base a los datos
tratados y la tarea asignada para el modelo. Estos pueden ser valores en
particular de los cuales se puede tener una idea de rangos posibles (e.g. taza de
aprendizaje para el algoritmo de optimización) o elecciones posibles de la
arquitectura del modelo (e.g. función de activación de cada capa).

La configuración elegida es crucial para que la optimización resulte
en un modelo preciso para la tarea asignada,y en el caso de los hiperparámetros
elegir un valor determinado puede ser difícil especialmente cuando son sensibles.
Una forma asistida para realizar esto es implementar un algoritmo de búsqueda
de parámetros el cual realiza elecciones particulares tomando muestras sobre
los rangos posibles determinados, así luego se optimiza un modelo por cada
configuración especificada. Opcionalmente, también se puede utilizar
validación cruzada para estimar la generalización del modelo obtenido con la
configuración y su independencia del conjunto de datos tomado.

Resumiendo, una búsqueda consta de:

* Un modelo estimador.
* Un espacio de parámetros.
* Un método para muestrear o elegir candidatos.
* Una función de evaluación para el modelo.
* (Opcional) Un esquema de validación cruzada.

"""

__author__ = 'leferrad'

from learninspy.core.model import NetworkParameters, NeuralNetwork
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.fileio import get_logger

import numpy as np

import os

logger = get_logger(name=__name__)
logger.propagate = False  # Para que no se dupliquen los mensajes por herencia


# optimization_domain = {}  # TODO Soportar esta funcionalidad
network_domain = {'n_layers': ((3, 7), 1),  # Teniendo en cuenta capa de entrada y salida
                  'activation': ['Tanh', 'ReLU', 'Softplus'],  # Tambien puede ponerse fun_activation.keys()
                  'dropout_ratios': ((0.0, 0.7), 1),  # ((begin, end), precision)
                  'l1': ((1e-6, 1e-4), 6), 'l2': ((1e-6, 1e-3), 4),  # ((begin, end), precision)
                  'perc_neurons': ((0.4, 1.5), 2)
                  }


class RandomSearch(object):
    """
    Una forma que evade buscar exahustivamente sobre el espacio de parámetros
    (lo cual es potencialmente costoso si dicho espacio es de una dimensión alta),
    es la de muestrear una determinada cantidad de veces el espacio, en forma
    aleatoria y no sobre una grilla determinada. Este método se denomina "búsqueda
    aleatoria" o *random search*, el cual es fácil de implementar como el *grid search*
    aunque se considera más eficiente especialmente en espacios de gran dimensión [bergstra2012random]_.

    En esta implementación, se deben especificar los parámetros específicos que se quieren explorar.
    Esto se realiza utilizando como medio la clase :class:`~learninspy.core.model.NetworkParameters`,
    en la cual se indica con un bool (True o False) sobre cada parámetro que se desea contemplar
    en la búsqueda de parámetros.
    También se pueden especificar los rangos o dominio de búsqueda (e.g. funciones de activación, cant.
    de capas y unidades en c/u, rangos de constantes para normas L1/L2, etc). Por defecto, se utiliza
    el dict 'network_domain' implementado en este módulo.

    :param net_params: :class:`~learninspy.core.model.NetworkParameters`
    :param n_layers: int, si es -1 se muestrea la cant. de capas, si es 0 se mantiene intacta la config,
     y si es > 0 representa la cant. de capas deseada.
    :param n_iter: int, cant. de iteraciones para la búsqueda.
    :param net_domain: dict, si es None se utiliza el dict 'network_domain' implementado en este módulo.
    :param seed: int, semilla que alimenta los generadores de números aleatorios.

    >>> from learninspy.core.model import NetworkParameters, NeuralNetwork
    >>> from learninspy.core.search import network_domain
    >>> net_params = NetworkParameters(units_layers=[4, 10, 3], activation=False, dropout_ratios=True,\ ...
    >>>                                classification=True, strength_l1=True, strength_l2=True, seed=123)
    >>> rnd_search = RandomSearch(net_params, n_layers=0, n_iter=10, net_domain=network_domain, seed=123)
    >>> rnd_search.fit(NeuralNetwork, train, valid, test)
    >>> ...

    **Referencias**:

    .. [bergstra2012random] Bergstra, J., & Bengio, Y. (2012).
                            Random search for hyper-parameter optimization.
                            Journal of Machine Learning Research, 13(Feb), 281-305.
    """

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
                units = int(p * units_layers[-1])
                if units <= 1:  # Verificar que haya una lista válida de neuronas
                    units = 2
                units_layers.append(units)
            units_layers.append(n_out)  # Por ultimo la capa de salida
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

    def fit(self, type_model, train, valid, test, mini_batch=100, parallelism=4, valid_iters=5, measure=None,
            stops=None, optimizer_params=None, reproducible=False, keep_best=True):
        """
        Función para iniciar la búsqueda de parámetros ajustada a las especificaciones de dominio dadas,
        utilizando los conjuntos de datos ingresados y demás parámetros de optimización para usar
        en la función de modelado :func:`~learninspy.core.model.NeuralNetwork.fit` en
        :class:`~learninspy.core.model.NeuralNetwork`.

        :param type_model: class, correspondiente a un tipo de modelo del módulo :mod:`~learninspy.core.model`.

        .. note:: El resto de los parámetros son los mismos que recibe la función
            :func:`~learninspy.core.model.NeuralNetwork.fit` incluyendo también el conjunto de prueba *test*
            que se utiliza para validar la conveniencia de cada modelo logrado.
            Remitirse a la API de dicha función para encontrar información de los parámetros.

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
            model = type_model(net_params_sample)
            hits_valid = model.fit(train, valid, mini_batch=mini_batch, parallelism=parallelism,
                                   valid_iters=valid_iters, measure=measure, reproducible=reproducible,
                                   stops=stops, optimizer_params=optimizer_params, keep_best=keep_best)
            hits_test = model.evaluate(test, predictions=False)
            if hits_test >= best_hits:
                best_hits = hits_test
                best_model = model
        logger.info("Configuracion del mejor modelo: %s", os.linesep+str(best_model.params))
        logger.info("Hits en test: %12.11f", best_hits)
        return best_model, best_hits