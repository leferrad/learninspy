#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np

# Librerias de Learninspy
from learninspy.utils.evaluation import RegressionMetrics
from learninspy.utils.data import label_data
from learninspy.core.model import NeuralNetwork, NetworkParameters, RegressionLayer, ClassificationLayer
from learninspy.utils.fileio import get_logger

# Librerias de Python
import copy

logger = get_logger(name=__name__)
logger.propagate = False  # Para que no se dupliquen los mensajes por herencia


class AutoEncoder(NeuralNetwork):
    """
    Tipo de red neuronal, compuesto de una capa de entrada, una oculta, y una de salida.
    Las unidades en la capa de entrada y la de salida son iguales, y en la capa oculta
    se entrena una representación de la entrada en distinta dimensión, mediante aprendizaje
    no supervisado y backpropagation..
    A las conexiones entre la capa de entrada y la oculta se le denominan **encoder**,
    y a las de la oculta a la salida se les llama **decoder**.

    Para más información, ver http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity

    :param params: model.NeuralNetworkParameters, donde se especifica la configuración de la red.
    :param list_layers: list de model.NeuralLayer, en caso de usar capas ya inicializadas.
    :param dropout_in: radio de DropOut usado para el encoder (el decoder no debe sufrir DropOut).

    >>> ae_params = NetworkParameters(units_layers=[5,3,5], activation='Tanh', dropout_ratios=None, classification=False)
    >>> ae = AutoEncoder(ae_params)
    """
    def __init__(self, params=None, list_layers=None, dropout_in=0.0):
        # TODO: incluir cuando haya SparseAutoencoder
        #self.sparsity_beta = 0
        #self.sparsity_param = 0.05

        # Aseguro algunos parametros
        params.classification = False
        n_in = params.units_layers[0]
        # params.activation[0] = 'Identity'

        params.units_layers.append(n_in)  # Unidades en la salida en igual cantidad que la entrada
        params.dropout_ratios = [dropout_in, 0.0]  # Dropout en encoder, pero nulo en decoder
        self.num_layers = 2
        NeuralNetwork.__init__(self, params, list_layers)


    # Override del backpropagation, para que sea de una sola capa oculta (TODO que incluya sparsity)
    def _backprop(self, x, y):
        # y es el label del aprendizaje supervisado. lo omito
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
        return cost, (nabla_w, nabla_b)

    def evaluate(self, data, predictions=False):
        """
        Evalúa AutoEncoder sobre un conjunto de datos.
        Se utiliza :math:`r^2` como métrica en la evaluación.

        :param data: list de LabeledPoint
        :param predictions: si es True, retorna las predicciones (salida del AutoEncoder)
        :return: resultado de evaluación, y predicciones si se solicita en *predictions*
        """
        actual = map(lambda lp: lp.features, data)  # Tiene que aprender a reconstruir la entrada
        predicted = map(lambda lp: self.predict(lp.features).matrix.T, data)  # TODO notar que tuve q transponer
        metric = RegressionMetrics(zip(predicted, actual))
        hits = metric.r2()
        if predictions is True:  # Devuelvo ademas el vector de predicciones
            ret = hits, predicted
        else:
            ret = hits
        return ret

    def _kl_divergence(self, x):
        raise NotImplementedError("Implementar para SparseAutoencoder!")

    def encode(self, x):
        """
        Codifica la entrada **x**, transformando los datos al pasarlos por el *encoder*.
        """
        if isinstance(x, list):
            x = map(lambda lp: self.encode(lp.features).matrix, x)
        else:
            x = self.encoder_layer().output(x, grad=False)   # Solo la salida de la capa oculta
        return x

    def encoder_layer(self):
        """
        Devuelve la capa de *encoder*.
        """
        return self.list_layers[0]

    def assert_regression(self):
        """
        Se asegura que el *decoder* corresponda a una capa de regresión
        (que sea del tipo *model.RegressionLayer*).
        """
        if type(self.list_layers[-1]) is ClassificationLayer:
            layer = RegressionLayer()
            layer.__dict__ = self.list_layers[-1].__dict__.copy()
            self.list_layers[-1] = layer


class StackedAutoencoder(NeuralNetwork):
    """
    Estructura de red neuronal profunda, donde los pesos de cada capa son inicializados con los datos de entrenamiento
    mediante **autoencoders**.

    Para más información, ver http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders

    :param params: model.NeuralNetworkParameters, donde se especifica la configuración de la red.
    :param list_layers: list de model.NeuralLayer, en caso de usar capas ya inicializadas.
    :param dropout: radio de DropOut a utilizar en el *encoder* de cada :class:`.AutoEncoder`.
    """

    def __init__(self, params, list_layers=None, dropout=None):

        # TODO: incluir cuando haya SparseAutoencoder
        #self.sparsity_beta = 0
        #self.sparsity_param = 0.05

        self.params = params
        self.num_layers = len(params.units_layers)
        if dropout is None:
            dropout = [0.0] * self.num_layers
        self.dropout = dropout
        NeuralNetwork.__init__(self, params, list_layers=None)
        self._init_ae()  # Creo autoencoders que se guardan en list_layers

    def _init_ae(self):  # TODO: cambiar nombre (no "ae")
        for l in xrange(self.num_layers - 1):
            # Genero nueva estructura de parametros acorde al Autoencoder a crear
            params = NetworkParameters(self.params.units_layers[l:l+2], activation=self.params.activation[l], # TODO: ojo si activation es una lista
                                        layer_distributed=self.params.layer_distributed, dropout_ratios=None,
                                        classification=False, strength_l1=self.params.strength_l1,
                                        strength_l2=self.params.strength_l2)
            self.list_layers[l] = AutoEncoder(params=params, dropout_in=self.dropout[l])
        # Configuro y creo la capa de salida (clasificación o regresión)
        params = NetworkParameters(self.params.units_layers[-2:], activation=self.params.activation,
                                        layer_distributed=self.params.layer_distributed, dropout_ratios=[0.0], # en salida no debe haber DropOut
                                        classification=self.params.classification, strength_l1=self.params.strength_l1,
                                        strength_l2=self.params.strength_l2)
        self.list_layers[-1] = NeuralNetwork(params=params)

    def fit(self, train, valid=None, stops=None, mini_batch=50, parallelism=4, optimizer_params=None,
            keep_best=False):
        """
        Fit de cada autoencoder usando conjuntos de entrenamiento y validación,
        y su apilado para entrenar la red neuronal profunda con aprendizaje no supervisado.
        Se especifica además cómo debe realizarse la optimización, mediante los parámetros explicados
        en el método :func:`~learninspy.core.model.NeuralNetwork.fit` de :class:`.NeuralNetwork`.

        :param train:
        :param valid:
        :param stops:
        :param mini_batch:
        :param parallelism:
        :param optimizer_params:
        :param keep_best:
        :return:
        """
        # Entreno Autoencoders
        train_ae = train
        valid_ae = valid
        labels_train = map(lambda lp: lp.label, train_ae)
        labels_valid = map(lambda lp: lp.label, valid_ae)
        for l in xrange(len(self.list_layers[:-1])):
            # Extraigo AutoEncoder
            ae = self.list_layers[l]
            logger.info("Entrenando AutoEncoder -> In: %i, Hidden: %i",
                        ae.params.units_layers[0], ae.params.units_layers[1])
            ae.assert_regression()  # Aseguro que sea de regresion (no puede ser de clasificacion)
            ae.fit(train_ae, valid_ae, stops=stops, mini_batch=mini_batch, parallelism=parallelism,
                   optimizer_params=optimizer_params, keep_best=keep_best)
            # Siguen siendo importantes los labels para el sample balanceado por clases
            train_ae = label_data(ae.encode(train_ae), labels_train)
            valid_ae = label_data(ae.encode(valid_ae), labels_valid)
            # Devuelvo AE a la lista
            self.list_layers[l] = copy.deepcopy(ae)

        # Se entrena tambien la capa de salida
        out_layer = self.list_layers[-1]
        logger.info("Entrenando Capa de salida -> In: %i, Out: %i",
                    out_layer.params.units_layers[0], out_layer.params.units_layers[1])
        out_layer.fit(train_ae, valid_ae, stops=stops, mini_batch=mini_batch, parallelism=parallelism,
                      optimizer_params=optimizer_params, keep_best=keep_best)
        self.list_layers[-1] = copy.deepcopy(out_layer)

        self.hits_valid = self.evaluate(valid)
        return self.hits_valid

    def finetune(self, train, valid, criterions=None, mini_batch=50, parallelism=4, optimizer_params=None,
                 keep_best=False):
        list_layers = copy.deepcopy(self.list_layers)
        list_layers[:-1] = map(lambda ae: ae.encoder_layer(), list_layers[:-1])  # Tomo solo la capa de encoder de cada ae
        list_layers[-1] = list_layers[-1].list_layers[0]  # Agarro la primer capa de la red que se genero para la salida
        nn = NeuralNetwork(self.params, list_layers=list_layers)
        hits_valid = nn.fit(train, valid, mini_batch=mini_batch, parallelism=parallelism, stops=criterions,
                            optimizer_params=optimizer_params)
        for l in xrange(len(self.list_layers) - 1):
            # Copio capa con ajuste fino al autoencoder
            self.list_layers[l].list_layers[0] = nn.list_layers[l]  # TODO mejorar esto, para que sea mas legible
        self.list_layers[-1].list_layers[0] = nn.list_layers[-1]
        return hits_valid

    def predict(self, x):
        if isinstance(x, list):
            x = map(lambda lp: self.predict(lp.features).matrix(), x)
        else:
            for i in xrange(self.num_layers-1):
                x = self.list_layers[i].encode(x)
            x = self.list_layers[-1].predict(x)
        return x

    # TODO hacer un override del plotter para graficar los weights y bias
