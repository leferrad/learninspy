#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Este módulo provee las clases utilizadas para modelar redes neuronales relacionadas a autoencoders
(e.g. AutoEncoder, StackerAutoencoder).
Es por ello que se corresponde a un caso o herencia del módulo principal :mod:`~learninspy.core.model`,
donde se sobrecargan las funcionalidades de dicho módulo para adaptarlas al diseño de autoencoders.
"""

__author__ = 'leferrad'

from learninspy.utils.evaluation import RegressionMetrics
from learninspy.utils.data import label_data, LabeledDataSet, DistributedLabeledDataSet
from learninspy.core.model import NeuralNetwork, NetworkParameters, RegressionLayer, ClassificationLayer
from learninspy.utils.fileio import get_logger

import numpy as np

import copy
import time

logger = get_logger(name=__name__)
logger.propagate = False  # Para que no se dupliquen los mensajes por herencia


class AutoEncoder(NeuralNetwork):
    """
    Tipo de red neuronal, compuesto de una capa de entrada, una oculta, y una de salida.
    Las unidades en la capa de entrada y la de salida son iguales, y en la capa oculta
    se entrena una representación de la entrada en distinta dimensión, mediante aprendizaje
    no supervisado y backpropagation.
    A las conexiones entre la capa de entrada y la oculta se le denominan **encoder**,
    y a las de la oculta a la salida se les llama **decoder**.

    Para más información, ver http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity

    :param params: model.NeuralNetworkParameters, donde se especifica la configuración de la red.
    :param list_layers: list de model.NeuralLayer, en caso de usar capas ya inicializadas.
    :param dropout_in: radio de DropOut usado para el encoder (el decoder no debe sufrir DropOut).

    >>> ae_params = NetworkParameters(units_layers=[5,3], activation='Tanh', dropout_ratios=None, classification=False)
    >>> ae = AutoEncoder(ae_params)
    """
    def __init__(self, params=None, list_layers=None, dropout_in=0.0):
        # Aseguro algunos parametros
        params.classification = False
        n_in = params.units_layers[0]
        params.layer_distributed.append(False)  # Ya que se agrega una dimension, se debe reflejar aqui tmb (False por ahora)

        params.units_layers.append(n_in)  # Unidades en la salida en igual cantidad que la entrada
        params.dropout_ratios = [dropout_in, 0.0]  # Dropout en encoder, pero nulo en decoder
        self.num_layers = 2
        NeuralNetwork.__init__(self, params, list_layers)

    #  Override del backpropagation, para que sea de una sola capa oculta
    def _backprop(self, x, y):
        # y es el label del aprendizaje supervisado. lo omito
        beg = time.time()  # tic
        num_layers = self.num_layers
        a = [None] * (num_layers + 1)  # Vector que contiene las activaciones de las salidas de cada NeuralLayer
        d_a = [None] * num_layers  # Vector que contiene las derivadas de las salidas activadas de cada NeuralLayer
        nabla_w = [None] * num_layers  # Vector que contiene los gradientes del costo respecto a W
        nabla_b = [None] * num_layers  # Vector que contiene los gradientes del costo respecto a b
        # Feed-forward
        a[0] = x  # Tomo como primer activacion la entrada x
        y = np.array(x)  # Debe aprender a reconstruir la entrada x
        for l in xrange(num_layers):
            (a[l + 1], d_a[l]) = self.list_layers[l].output(a[l], grad=True)
        cost = a[-1].loss(self.loss, y)
        # Backward pass
        d_cost = a[-1].loss_d(self.loss_d, y)
        delta = d_cost.mul_elemwise(d_a[-1])
        nabla_w[-1] = delta.outer(a[-2])
        nabla_b[-1] = delta
        for l in xrange(2, num_layers + 1):
            w_t = self.list_layers[-l + 1].weights.transpose()
            delta = w_t.mul_array(delta).mul_elemwise(d_a[-l])
            nabla_w[-l] = delta.outer(a[-l - 1])
            nabla_b[-l] = delta
        end = (time.time() - beg) * 1000.0  # toc (ms)
        logger.debug("Duration of computing gradients on backpropagation: %8.4fms.", end)
        return cost, (nabla_w, nabla_b)

    def evaluate(self, data, predictions=False, measure=None):
        """
        Evalúa el AutoEncoder sobre un conjunto de datos, lo que equivale a medir su desempeño en
        reconstruir dicho conjunto de entrada.

        :param data: list de LabeledPoint o instancia de :class:`~learninspy.utils.data.LabeledDataSet`.
        :param predictions: si es True, retorna las predicciones (salida reconstruida por el AutoEncoder).
        :param measure: string, key de alguna medida implementada en
         :class:`~learninspy.utils.evaluation.RegressionMetrics`.
        :return: resultado de evaluación, y predicciones si se solicita en *predictions*
        """
        if isinstance(data, LabeledDataSet):
            actual = data.features
            if type(data) is DistributedLabeledDataSet:
                predicted = actual.map(lambda f: self.predict(f).matrix.T).collect()  # TODO: notar la transposic
                actual = actual.collect()
            else:
                predicted = map(lambda f: self.predict(f).matrix.T, actual)
        else:
            actual = map(lambda lp: lp.features, data)
            predicted = map(lambda f: self.predict(f).matrix.T, actual)

        metrics = RegressionMetrics(zip(predicted, actual))
        if measure is None:
            measure = 'R2'
        # Evaluo en base a la medida elegida (key perteneciente al dict 'metrics.measures')
        hits = metrics.evaluate(measure=measure)

        if predictions is True:  # Devuelvo ademas el vector de predicciones
            ret = hits, predicted
        else:
            ret = hits
        return ret

    def encode(self, x):
        """
        Codifica la entrada **x**, transformando los datos al pasarlos por el *encoder*.

        :param x: list of LabeledPoints.
        :return: list of numpy.ndarray
        """
        if isinstance(x, list):
            x = map(lambda lp: self.encode(lp.features), x)
        else:
            x = self.encoder_layer().output(x, grad=False).matrix   # Solo la salida de la capa oculta
            # 'x' es un vector columna ya que es la salida de una capa.
            x = x.ravel()  # Se quiere como vector fila, por lo cual el ravel hace el 'flattening'.
        return x

    def encoder_layer(self):
        """
        Devuelve la capa de neuronas correspondiente al *encoder*.
        """
        return self.list_layers[0]

    def _assert_regression(self):
        """
        Se asegura que el *decoder* corresponda a una capa de regresión
        (que sea del tipo *model.RegressionLayer*).
        """
        if type(self.list_layers[-1]) is ClassificationLayer:
            layer = RegressionLayer(n_in=2, n_out=2)  # Inicializo con basura, luego se sobreescribe
            layer.__dict__ = self.list_layers[-1].__dict__.copy()
            self.list_layers[-1] = layer


# TODO: no debería ser StackedAutoEncoder?
class StackedAutoencoder(NeuralNetwork):
    """
    Estructura de red neuronal profunda, donde los parámetros de cada capa son inicializados con los datos de entrenamiento
    mediante instancias de :class:`~learninspy.core.autoencoder.AutoEncoder`.

    Para más información, ver http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders

    :param params: :class:`~learninspy.core.model.NetworkParameters`, donde se especifica la configuración de la red.
    :param list_layers: list de :class:`~learninspy.core.model.NeuralLayer`, en caso de querer usar capas ya inicializadas.
    :param dropout: ratio de DropOut a utilizar en el *encoder* de cada :class:`.AutoEncoder`.
    """

    def __init__(self, params, list_layers=None, dropout=None):
        self.params = params
        self.num_layers = len(params.units_layers)
        if dropout is None:
            dropout = [0.0] * self.num_layers
        self.dropout = dropout
        NeuralNetwork.__init__(self, params, list_layers=None)
        self._init_autoencoders()  # Creo autoencoders que se guardan en list_layers

    def _init_autoencoders(self):
        """
        Inicialización de los parámetros de cada autoencoder que conforman esta red.

        :return:
        """
        for l in xrange(self.num_layers - 1):
            # Genero nueva estructura de parametros acorde al Autoencoder a crear
            params = NetworkParameters(self.params.units_layers[l:l+2], activation=self.params.activation[l],  # TODO: ojo si activation es una lista
                                       layer_distributed=self.params.layer_distributed, dropout_ratios=None,
                                       classification=False, strength_l1=self.params.strength_l1,
                                       strength_l2=self.params.strength_l2)
            self.list_layers[l] = AutoEncoder(params=params, dropout_in=self.dropout[l])
        # Configuro y creo la capa de salida (clasificación o regresión)
        params = NetworkParameters(self.params.units_layers[-2:], activation=self.params.activation,
                                   layer_distributed=self.params.layer_distributed, dropout_ratios=[0.0],  # en salida no debe haber DropOut
                                   classification=self.params.classification, strength_l1=self.params.strength_l1,
                                   strength_l2=self.params.strength_l2)
        self.list_layers[-1] = NeuralNetwork(params=params)

    # TODO: renombrar a 'pretrain' y que la función 'fit' llame a esta fun y luego a 'finetune'
    def fit(self, train, valid=None, mini_batch=50, parallelism=4, valid_iters=10, measure=None,
            stops=None, optimizer_params=None, reproducible=False, keep_best=False):
        """
        Fit de cada autoencoder usando conjuntos de entrenamiento y validación,
        y su apilado para pre-entrenar la red neuronal profunda con aprendizaje no supervisado.
        Finalmente se entrena un clasificador Softmax sobre la salida del último autoencoder
        entrenado.

        .. note:: Dado que esta función es una sobrecarga del método original
           :func:`~learninspy.core.model.NeuralNetwork.fit`, se puede remitir a la documentación
           de esta última para conocer el significado de los parámetros.
        """
        # Entreno Autoencoders
        train_ae = train
        valid_ae = valid
        train = None  # Dejo en None ya que no se usa más. TODO: está bien?

        labels_train = map(lambda lp: lp.label, train_ae)
        labels_valid = map(lambda lp: lp.label, valid_ae)
        for l in xrange(len(self.list_layers[:-1])):
            # Extraigo AutoEncoder
            ae = self.list_layers[l]
            logger.info("Entrenando AutoEncoder -> In: %i, Hidden: %i",
                        ae.params.units_layers[0], ae.params.units_layers[1])
            ae._assert_regression()  # Aseguro que sea de regresion (no puede ser de clasificacion)
            ae.fit(train_ae, valid_ae, valid_iters=valid_iters, stops=stops, mini_batch=mini_batch,
                   parallelism=parallelism, measure=measure, optimizer_params=optimizer_params,
                   reproducible=reproducible, keep_best=keep_best)
            # Siguen siendo importantes los labels para el sample balanceado por clases
            train_ae = label_data(ae.encode(train_ae), labels_train)
            valid_ae = label_data(ae.encode(valid_ae), labels_valid)
            # Devuelvo AE a la lista
            self.list_layers[l] = copy.deepcopy(ae)

        # Se entrena tambien la capa de salida
        out_layer = self.list_layers[-1]
        logger.info("Entrenando Capa de salida -> In: %i, Out: %i",
                    out_layer.params.units_layers[0], out_layer.params.units_layers[1])
        out_layer.fit(train_ae, valid_ae, valid_iters=valid_iters, stops=stops, mini_batch=mini_batch,
                      parallelism=parallelism, measure=measure, optimizer_params=optimizer_params,
                      reproducible=reproducible, keep_best=keep_best)
        self.list_layers[-1] = copy.deepcopy(out_layer)
        # Evaluación final (copio resultados de última capa)
        self.hits_train = out_layer.hits_train
        self.hits_valid = out_layer.hits_valid
        self.epochs = out_layer.epochs
        return self.hits_valid[-1]

    def finetune(self, train, valid, mini_batch=50, parallelism=4, valid_iters=10, measure=None,
                 stops=None,  optimizer_params=None, reproducible=False, keep_best=False):
        """
        Ajuste fino con aprendizaje supervisado de la red neuronal, cuyos parámetros fueron inicializados mediante
        el pre-entrenamiento de los autoencoders.

        .. note:: Dado que esta función es una sobrecarga del método original
           :func:`~learninspy.core.model.NeuralNetwork.fit`, se puede remitir a la documentación
           de esta última para conocer el significado de los parámetros.
        """
        list_layers = copy.deepcopy(self.list_layers)
        list_layers[:-1] = map(lambda ae: ae.encoder_layer(), list_layers[:-1])  # Tomo solo la capa de encoder de cada ae
        list_layers[-1] = list_layers[-1].list_layers[0]  # Agarro la primer capa de la red que se genero para la salida
        nn = NeuralNetwork(self.params, list_layers=list_layers)
        hits_valid = nn.fit(train, valid, mini_batch=mini_batch, parallelism=parallelism,
                            valid_iters=valid_iters, measure=measure, stops=stops, keep_best=keep_best,
                            optimizer_params=optimizer_params, reproducible=reproducible)
        for l in xrange(len(self.list_layers) - 1):
            # Copio capa con ajuste fino al autoencoder
            self.list_layers[l].list_layers[0] = nn.list_layers[l]  # TODO mejorar esto, para que sea mas legible
        self.list_layers[-1].list_layers[0] = nn.list_layers[-1]
        # Copio los resultados de evaluación también (sobrescribir en lugar de guardar dos listas más)
        self.hits_train = nn.hits_train
        self.hits_valid = nn.hits_valid
        self.epochs = nn.epochs
        return hits_valid

    def predict(self, x):
        """
        Predicciones sobre una entrada de datos (singular o conjunto).

        :param x: *numpy.ndarray* o *pyspark.mllib.regression.LabeledPoint*, o list de ellos.
        :return: *numpy.ndarray*
        """
        beg = time.time()  # tic
        if isinstance(x, list):
            x = map(lambda lp: self.predict(lp.features).matrix(), x)
        else:
            for i in xrange(self.num_layers-1):
                x = self.list_layers[i].encode(x)
            x = self.list_layers[-1].predict(x)
        end = (time.time() - beg) * 1000.0  # toc (ms)
        logger.debug("Duration of computing predictions to produce output : %8.4fms.", end)
        return x
