#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Este es el módulo principal del framework, donde se proveen las clases referidas al modelado de redes neuronales.
El mismo consta de clases para crear una red neuronal, su composición de capas de neuronas, y la configuración
de la misma mediante la especificación de parámetros.
"""

__author__ = 'leferrad'

from learninspy.core import activations as act, loss, optimization as opt
from learninspy.core.stops import criterion
from learninspy.core.neurons import LocalNeurons
from learninspy.utils.evaluation import ClassificationMetrics, RegressionMetrics
from learninspy.utils.data import LabeledDataSet, DistributedLabeledDataSet, label_to_vector
from learninspy.context import sc
from learninspy.utils.fileio import get_logger

import copy
import cPickle as pickle
import os
import time
import gc

import numpy as np
from pyspark import StorageLevel
from pyspark.mllib.regression import LabeledPoint


logger = get_logger(name=__name__)
logger.propagate = False  # Para que no se dupliquen los mensajes por herencia


class NeuralLayer(object):
    """
    Clase básica para modelar una capa de neuronas que compone una red neuronal.
    Contiene sus "neuronas" representadas por pesos sinápticos **w** y **b**,
    además de una función de activación asociada para dichos pesos.

    Una correcta inicialización de los pesos sinápticos está muy ligada a la función de activación elegida:

    * Por defecto, los pesos sinápticos se inicializan con una distribución uniforme \
        con media *0* y varianza :math:`\\tfrac{2.0}{\sqrt{n_{in}}}`, \
        lo cual da buenos resultados especialmente usando ReLUs.
    * Para la función *Tanh* se muestrea sobre una distribución uniforme \
        en el rango :math:`\pm \sqrt{\\frac{6}{n_{in}+n_{out}}}`.
    * Para la *Sigmoid* en el rango :math:`\pm 4.0 \sqrt{\\frac{6}{n_{in}+n_{out}}}`.

    :param n_in: int, dimensión de la entrada.
    :param n_out: int, dimensión de la salida.
    :param activation: string, key de alguna función de activación soportada en :mod:`~learninspy.core.activations`.
    :param distributed: si es True, indica que se utilicen arreglos distribuidos para **w** y **b**.
    :param w: :class:`.LocalNeurons`, matriz de pesos sinápticos. Si es *None*, se crea por defecto.
    :param b: :class:`.LocalNeurons`, vector de pesos bias. Si es *None*, se crea por defecto.
    :param rng: si es *None*, se crea un generador de números aleatorios
     mediante una instancia **numpy.random.RandomState**.

    .. note:: el parámetro *distributed* no tiene efecto, ya que el uso de arreglos distribuidos
       se deja para un próximo release.

    >>> n_in, n_out = (10, 5)
    >>> layer = NeuralLayer(n_in, n_out, activation='Tanh')
    >>> x = np.random.rand(n_in)
    >>> out = layer.output(x)
    >>> len(out)
    5

    """

    def __init__(self, n_in, n_out, activation='ReLU', distributed=False, w=None, b=None, rng=None):
        self.n_out = n_out
        self.n_in = n_in
        self.activation = act.fun_activation[activation]
        self.activation_d = act.fun_activation_d[activation]
        distributed = False  # TODO completar esta funcionalidad

        if rng is None:
            rng = np.random.RandomState(123)
        self.rng = rng
        self.rnd_state = self.rng.get_state()

        self.shape_w = n_out, n_in
        self.shape_b = n_out, 1

        # Recomendaciones de http://cs231n.github.io/neural-networks-2/ y http://deeplearning.net/tutorial/mlp.html#mlp
        # TODO: ver si conviene dejar acá la inicializ de pesos, o en core.neurons (en términos de legibilidad)
        if w is None:
            if activation is "Tanh":
                w = np.asarray(
                    self.rng.uniform(
                        low=-np.sqrt(6.0 / (n_in + n_out)),
                        high=+np.sqrt(6.0 / (n_in + n_out)),
                        size=self.shape_w),
                    dtype=np.dtype(float)
                )
            elif activation is "Sigmoid":
                w = np.asarray(
                    self.rng.uniform(
                        low=-np.sqrt(6.0 / (n_in + n_out))*4.0,
                        high=+np.sqrt(6.0 / (n_in + n_out))*4.0,
                        size=self.shape_w),
                    dtype=np.dtype(float)
                )
            else:
                w = self.rng.randn(*self.shape_w) * np.sqrt(2.0/n_in)

        if b is None:
            b = np.zeros(self.shape_b, dtype=np.dtype(float))

        # TODO weights_T era p/ poder hacer operaciones distribuidas, pero se deja como TBC la class DistributedNeurons
        assert distributed is False, logger.error("DistributedNeurons will be implemented soon ...")
        self.weights = LocalNeurons(w, self.shape_w)
        #self.weights_T = LocalNeurons(w.transpose(), self.shape_w[::-1])
        self.bias = LocalNeurons(b, self.shape_b)

    def __div__(self, other):
        self.weights /= other
        self.bias /= other
        return self

    def __mul__(self, other):
        self.weights *= other
        self.bias *= other
        return self

    def l1(self):
        """
        Norma **L1** sobre la matriz **w** de pesos sinápticos,
        utilizando la funcion :func:`~learninspy.core.neurons.LocalNeurons.l1`.

        Por lo tanto, se retorna el resultado de aplicar la norma y el gradiente de la misma.

        :return: tuple de float, :class:`~learninspy.core.neurons.LocalNeurons`
        """
        return self.weights.l1()

    def l2(self):
        """
        Norma **L2** sobre la matriz **w** de pesos sinápticos,
        utilizando la funcion :func:`~learninspy.core.neurons.LocalNeurons.l2`.

        Por lo tanto, se retorna el resultado de aplicar la norma y el gradiente de la misma.

        :return: tuple de float, :class:`~learninspy.core.neurons.LocalNeurons`
        """
        return self.weights.l2()

    def output(self, x, grad=False):
        """
        Salida de la capa neuronal. Se toma una entrada :math:`x \in \Re^{n_{in}}`, se pondera con los
        pesos sinápticos **W** y el bias **b**, y luego se aplica la función de activación **f** para retornar como
        resultado:

        :math:`a = f(Wx + b), \quad a' = f'(Wx + b)`

        :param x: **numpy.ndarray**, vector de entrada
        :param grad: Si es *True*, se retorna además el gradiente de la salida.
        :return: **numpy.ndarray**, o tupla de ellos si *grad* es True.
        """
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.activation(self.activation)
        if grad is True:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)
        return a

    # Basado en http://cs231n.github.io/neural-networks-2/
    def dropoutput(self, x, p, grad=True):
        """
        Salida de la capa neuronal, luego de aplicar la regularización de los pesos sinápticos por Dropout
        utilizando la funcion :func:`~learninspy.core.neurons.LocalNeurons.dropout`.

        :param x: numpy.ndarray, vector de entrada
        :param p: float, tal que :math:`0<p<1`
        :param grad: Si es *True*, se retorna además el gradiente de la salida.
        :return: numpy.ndarray, (o tuple de  numpy.ndarray, numpy.ndarray si *grad* es *True*),
         numpy.ndarray correspondiente a la máscara binaria utilizada en el DropOut.

        .. note:: En las predicciones de la red no se debe efectuar Dropout.
        """
        self.rng.set_state(self.rnd_state)  # Para que sea reproducible
        out = self.output(x, grad)
        if grad is True:
            a, d_a = out
            a, mask = a.dropout(p, self.rng.randint(500))  # randint es para generar un nuevo seed (reproducible)
            out = a, d_a
        else:
            out, mask = out.dropout(p, self.rng.randint(500))  # TODO: en que caso no se necesitaria el grad?
        return out, mask  # Devuelvo ademas la mascara utilizada, para el backprop

    def update(self, step_w, step_b):  # Actualiza sumando los argumentos w y b a los respectivos pesos
        """
        Se actualizan los arreglos **w** y **b** sumando respectivamente los incrementos
        dados por los parámetros recibidos.

        :param step_w: :class:`.LocalNeurons`
        :param step_b: :class:`.LocalNeurons`
        """
        self.weights += step_w
#        self.weights_T += step_w.transpose()
        self.bias += step_b
        return

    def get_weights(self):
        """
        Se devuelve la matriz de pesos sinápticos **w**.

        :return: numpy.ndarray.
        """
        return self.weights

    def get_bias(self):
        """
        Se devuelve el vector de bias **b**.

        :return: numpy.ndarray.
        """
        return self.bias


class ClassificationLayer(NeuralLayer):
    """
    Clase correspondiente a la capa de salida en una red neuronal con tareas de clasificación.
    Se distingue de una :class:`.RegressionLayer` en que para realizar la clasificación se define
    que la activación se de por la función *softmax*.

    """

    def output(self, x, grad=False):
        """
        Salida de la capa de clasificación. Similar a la función de la clase madre
        :func:`~learninspy.core.model.NeuralLayer.output`, pero la activación está dada
        por la función Softmax utilizando el método :func:`~learninspy.core.neurons.LocalNeurons.softmax`
        para efectuar la clasificación deseada sobre la entrada *x*.

        Dado que el gradiente del Softmax está computado en la función por la regla de la cadena,
        se omite el cómputo aquí y por ende el parámetro *grad* es innecesario
        aunque persiste para tener compatibilidad con el resto del esquema.

        :param x: **numpy.ndarray**, vector de entrada
        :param grad: bool
        :return: **numpy.ndarray**, o tupla de ellos si *grad* es True.

        .. note:: si *grad* es True, sólo se retorna un vector "basura" que no es utilizado en el backpropagation.
        """
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.softmax()  # La activacion es un clasificador softmax
        if grad is True:  # Dado que el gradiente está cubierto por el loss de CE, no se necesita computar aquí
            a = (a, a)  # Por lo tanto, se retorna un vector "basura" para no romper el esquema del backpropagation
        return a

    def dropoutput(self, x, p, grad=False):
        """
        .. warning:: No se debe aplicar Dropout en la capa de salida de una red neuronal,
            por lo cual este método arroja un error de excepción.
        """
        raise Exception("Don't use dropout for output layer")


class RegressionLayer(NeuralLayer):
    """
    Clase correspondiente a la capa de salida en una red neuronal con tareas de regresión,
    utilizando la función de activación como salida de la red.

    .. note:: No es recomendado utilizar Dropout en las capas de una red neuronal con tareas de regresión.
    """
    #  Importante leer 'Word of caution' en http://cs231n.github.io/neural-networks-2/
    #  El output es el mismo que una NeuralLayer
    def dropoutput(self, x, p, grad=False):
        """

        .. warning:: No se debe aplicar Dropout en la capa de salida de una red neuronal,
            por lo cual este método arroja un error de excepción.
        """
        raise Exception("Don't use dropout for output layer")


class NetworkParameters:
    """
    Clase utilizada para especificar todos los parámetros necesarios para configurar una red neuronal

    :param units_layers: list of ints, donde cada valor indica la cantidad de unidades
        que posee la respectiva capa. La cantidad de valores de la lista indica el total
        de capas que va a tener la red (entrada + ocultas + salida).
    :param activation: string or list of strings, indicando la key de la/s activación/es a utilizar en
        las capas de la red neuronal.
    :param layer_distributed: list of bools, indicando por cada capa si sus neuronas van a representarse
        o no por arreglos distribuidos (**no tiene efecto en este release**).
    :param dropout_ratios: list of floats, indicando el valor de *p* para aplicar Dropout en cada
        respectiva capa.
    :param classification: bool, es *True* si la tarea de la red es de clasificación y *False*
        si es de regresión.
    :param strength_l1: float, ratio de Norma **L1** a aplicar en todas las capas.
    :param strength_l2: float, ratio de Norma **L2** a aplicar en todas las capas.
    :param seed: int, semilla que alimenta al generador de números aleatorios **numpy.random.RandomState**
        utilizado por la red.

    >>> net_params = NetworkParameters(units_layers=[4, 8, 3], dropout_ratios=[0.0, 0.0],\ ...
    >>>                                activation='ReLU', strength_l1=1e-5, strength_l2=3e-4,\ ...
    >>>                                classification=True, seed=123)
    >>> net_params == net_params
    True
    >>> net_params == NetworkParameters(units_layers=[10, 2])
    False
    >>> print str(net_params)
    Layer 0 with 4 neurons, using ReLU activation and 0.0 ratio of DropOut.
    Layer 1 with 8 neurons, using ReLU activation and 0.0 ratio of DropOut.
    Layer 2 with 3 neurons, using Softmax activation.
    The loss is CrossEntropy for a task of classification.
    L1 strength is 1e-05 and L2 strength is 0.0003.
    """
    def __init__(self, units_layers, activation='ReLU', dropout_ratios=None, layer_distributed=False,
                 classification=True, strength_l1=1e-5, strength_l2=1e-4, seed=123):
        num_layers = len(units_layers)  # Cant total de capas (entrada + ocultas + salida)
        if dropout_ratios is None:
            if classification is True:
                # Default, dropout de 0.2 en entrada, de 0.5 en ocultas y la salida no debe tener
                dropout_ratios = [0.2] + [0.5] * (num_layers-2) + [0.0]
            else:
                dropout_ratios = [0.0] * num_layers  # Nunca recomendado hacer dropout en regresion
            dropout_ratios[-1] = 0.0  # TODO es para asegurarme que no haya DropOut en la salida, pero verificar mejor
        if type(dropout_ratios) is list and len(dropout_ratios) < (num_layers - 1):
            dropout_ratios.append(0.0)  # Se completa la lista de ratios con un 0.0 para la salida

        if type(layer_distributed) is bool:  # Por defecto, las capas no estan distribuidas (default=False)
            layer_distributed = [layer_distributed] * num_layers

        if type(activation) is not list:  # Si es un string, lo replico por la cant de capas
            activation = [activation] * num_layers

        self.activation = activation
        self.dropout_ratios = dropout_ratios  # Recordar que la capa de salida no sufre dropout
        # 'units_layers' es un vector que indica la cantidad de unidades de cada capa (1er elemento se refiere a capa visible)
        self.units_layers = units_layers
        self.layer_distributed = layer_distributed
        self.classification = classification

        if classification is True:
            self.loss = 'CrossEntropy'  # Loss para clasificacion
        else:
            self.loss = 'MSE'  # Loss para regresion

        self.strength_l1 = strength_l1
        self.strength_l2 = strength_l2
        self.rng = np.random.RandomState(seed)

    def __eq__(self, other):
        assert isinstance(other, NetworkParameters), ValueError("Se necesita una instancia de NetworkParameters")
        equal_values = []
        for k_self, v_self in self.__dict__.items():
            if k_self is not 'rng':  # No tiene sentido comparar objetos np.random.RandomState
                equals = other.__dict__[k_self] == v_self
                equal_values.append(equals)
        return all(equal_values)

    def __str__(self):
        config = ""
        for l in xrange(len(self.units_layers)):
            if l == (len(self.units_layers)-1):
                if self.classification is True:  # Para especificar softmax de clasific
                    config += "Layer "+str(l)+" with "+str(self.units_layers[l])+" neurons, using " \
                              + "Softmax activation."+os.linesep
                else:  # Regresion
                    config += "Layer "+str(l)+" with "+str(self.units_layers[l])+" neurons, using " \
                              + self.activation[l]+" activation."+os.linesep
            else:
                config += "Layer "+str(l)+" with "+str(self.units_layers[l])+" neurons, using " \
                          + self.activation[l]+" activation and "\
                          + str(self.dropout_ratios[l])+" ratio of DropOut."+os.linesep
        config += "The loss is "+self.loss+" for a task of "
        if self.classification is True:
            config += "classification."+os.linesep
        else:
            config += "regression."+os.linesep
        config += "L1 strength is "+str(self.strength_l1) + \
                  " and L2 strength is "+str(self.strength_l2)+"."+os.linesep
        return config


class NeuralNetwork(object):
    """
    Clase para modelar una red neuronal. La misma soporta funcionalidades para configuración y diseño,
    y para la optimización y testeo sobre un conjunto de datos cargado. Además ofrece funciones para
    cargar y guardar un modelo entrenado.

    :param params: :class:`.NetworkParameters`, parámetros que configuran la red.
    :param list_layers: list of :class:`.NeuralLayer`, en caso de que se utilicen capas de neuronas
        ya creadas.
    """
    def __init__(self, params, list_layers=None):
        self.params = params
        self.list_layers = list_layers  # En caso de que la red reciba capas ya inicializadas
        self.loss = loss.fun_loss[self.params.loss]
        self.loss_d = loss.fun_loss_d[self.params.loss]
        self.rnd_state = self.params.rng.get_state()

        if list_layers is None:
            self.list_layers = []  # Creo un arreglo vacio para ir agregando las capas que se inicializan
            self.__init_weights()
        else:
            # Me aseguro que la capa de salida sea acorde al problema en cuestion (dado por flag params.classification)
            self.__assert_type_outputlayer()
        self.num_layers = len(self.list_layers)
        self.hits_train = []
        self.hits_valid = []
        self.epochs = []

    def __assert_type_outputlayer(self):
        """
        Función interna creada para asegurar que la capa de salida sea correcta en cuanto a la tarea de la red.

        :return:
        """
        if self.params.classification is True:  # Problema de clasificacion
            if type(self.list_layers[-1]) is not ClassificationLayer:  # Capa de salida no es de clasificacion
                new_outputlayer = ClassificationLayer()
                new_outputlayer.__dict__.update(self.list_layers[-1].__dict__)
                self.list_layers[-1] = new_outputlayer  # Cambio tipo de capa de salida
        else:  # Problema de regresion
            if type(self.list_layers[-1]) is not RegressionLayer:
                new_outputlayer = RegressionLayer()
                new_outputlayer.__dict__.update(self.list_layers[-1].__dict__)
                self.list_layers[-1] = new_outputlayer  # Cambio tipo de capa de salida
        return

    def __init_weights(self):  # Metodo privado
        logger.debug("Initializing weights and bias of Neural Layers ...")
        num_layers = len(self.params.units_layers)
        self._set_initial_rndstate()
        for i in xrange(1, num_layers - 1):
            self.list_layers.append(NeuralLayer(self.params.units_layers[i - 1], self.params.units_layers[i],
                                                self.params.activation[i - 1],
                                                distributed=self.params.layer_distributed[i],
                                                rng=self.params.rng))
        if self.params.classification is True:
            # Ultima capa es de clasificacion, por lo que su activacion es softmax
            self.list_layers.append(ClassificationLayer(self.params.units_layers[-2],
                                                        self.params.units_layers[-1],
                                                        self.params.activation[-2],
                                                        # Notar que no tomo la ultima, ya que la lista tiene 1 elem mas
                                                        distributed=self.params.layer_distributed[-1],
                                                        rng=self.params.rng))
        else:
            # Ultima capa es de regresion, por lo que su activacion puede ser cualquiera
            # (la misma que las ocultas por default)
            self.list_layers.append(RegressionLayer(self.params.units_layers[num_layers - 2],
                                                    self.params.units_layers[num_layers - 1],
                                                    self.params.activation[-1],
                                                    # Notar que ahora tomo la ultima, ya que se corresponde a la salida
                                                    distributed=self.params.layer_distributed[num_layers - 1],
                                                    rng=self.params.rng))

    def _set_initial_rndstate(self):
        self.params.rng.set_state(self.rnd_state)
        return

    def l1(self):
        """
        Norma **L1** sobre la matriz **w** de pesos sinápticos de cada una de las N capas en la red,
        calculada mediante llamadas a la funcion :func:`~learninspy.core.neurons.LocalNeurons.l1`,
        tal que se obtiene:

        :math:`L1=\lambda_1 \displaystyle\sum\limits_{l}^N L_1(W^{l})`

        Además se retorna la lista de N gradientes correspondientes a cada capa de la red.

        :return: tuple de float, list de :class:`~learninspy.core.neurons.LocalNeurons`
        """
        cost = 0.0
        gradient = []
        for layer in self.list_layers:
            c_l1, g_l1 = layer.l1()
            cost += c_l1
            gradient.append(g_l1 * self.params.strength_l1)
        cost *= self.params.strength_l1
        return cost, gradient

    def l2(self):
        """
        Norma **L2** sobre la matriz **w** de pesos sinápticos de cada una de las N capas en la red,
        calculada mediante llamadas a la funcion :func:`~learninspy.core.neurons.LocalNeurons.l2`,
        tal que se obtiene:

        :math:`L2=\lambda_2 \displaystyle\sum\limits_{l}^N L_2(W^{l})`

        Además se retorna la lista de N gradientes correspondientes a cada capa de la red.

        :return: tuple de float, list de :class:`~learninspy.core.neurons.LocalNeurons`
        """
        cost = 0.0
        gradient = []
        for layer in self.list_layers:
            c_l2, g_l2 = layer.l2()
            cost += c_l2
            gradient.append(g_l2 * self.params.strength_l2)
        cost *= self.params.strength_l2
        return cost, gradient

    def set_l1(self, strength_l1):
        """
        Setea un strength dado para calcular la norma L1,
        sobrescribiendo el valor correspondiente en la instancia de parámetros
        :class:`~learninspy.core.model.NetworkParameters`.

        :param strength_l1: float
        """
        self.params.strength_l1 = strength_l1
        return

    def set_l2(self, strength_l2):
        """
        Setea un strength dado para calcular la norma L2,
        sobrescribiendo el valor correspondiente en la instancia de parámetros
        :class:`~learninspy.core.model.NetworkParameters`.

        :param strength_l2: float
        """
        self.params.strength_l2 = strength_l2
        return

    def set_dropout_ratios(self, dropout_ratios):
        """
        Setea los ratios para utilizar en el DropOut de los pesos sinápticos,
        sobrescribiendo el valor correspondiente en la instancia de parámetros
        :class:`~learninspy.core.model.NetworkParameters`.

        :param dropout_ratios: list de floats
        """
        self.params.dropout_ratios = dropout_ratios
        return

    def _backprop(self, x, y):
        """
        Algoritmo de backpropagation para ajustar la red neuronal con una entrada {x,y}

        :param x: *numpy.ndarray* (features)
        :param y: float o *numpy.ndarray* (label)
        :return: float (costo), tuple de :class:`~learninspy.core.neurons.LocalNeurons` (gradientes de W y b)
        """
        # Ver http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
        beg = time.time()  # tic
        num_layers = self.num_layers
        drop_fraction = self.params.dropout_ratios  # Vector con las fracciones de DropOut para cada NeuralLayer
        mask = [None] * num_layers  # Vector que contiene None o la mascara de DropOut, segun corresponda
        a = [None] * (num_layers + 1)  # Vector que contiene las activaciones de las salidas de cada NeuralLayer
        d_a = [None] * num_layers  # Vector que contiene las derivadas de las salidas activadas de cada NeuralLayer
        nabla_w = [None] * num_layers  # Vector que contiene los gradientes del costo respecto a W
        nabla_b = [None] * num_layers  # Vector que contiene los gradientes del costo respecto a b
        # Feed-forward
        a[0] = x  # Tomo como primer activacion la entrada x
        for l in xrange(num_layers):
            if drop_fraction[l] > 0.0:
                (a[l + 1], d_a[l]), mask[l] = self.list_layers[l].dropoutput(a[l], drop_fraction[l], grad=True)
            else:
                (a[l + 1], d_a[l]) = self.list_layers[l].output(a[l], grad=True)
        cost = a[-1].loss(self.loss, y)
        # Backward pass
        d_cost = a[-1].loss_d(self.loss_d, y)
        logger.debug("Loss on backpropagation: %10.3f", cost)
        # Si el problema es de clasificación, entonces se utiliza una Softmax y por ende
        # el gradiente ya está computado en d_cost por el loss de CE.
        if self.params.classification is True:
            delta = d_cost  # Ya incluye el producto con el gradiente del Softmax
        else:
            delta = d_cost.mul_elemwise(d_a[-1])
        if drop_fraction[-1] > 0.0:  # No actualizo las unidades "tiradas"
            delta = delta.mul_elemwise(mask[-1])
        nabla_w[-1] = delta.outer(a[-2])
        nabla_b[-1] = delta
        for l in xrange(2, num_layers + 1):
            w_t = self.list_layers[-l + 1].weights.transpose()
            delta = w_t.mul_array(delta).mul_elemwise(d_a[-l])
            if drop_fraction[-l] > 0.0:  # No actualizo las unidades "tiradas"
                delta = delta.mul_elemwise(mask[-l])
            nabla_w[-l] = delta.outer(a[-l - 1])
            nabla_b[-l] = delta
        end = (time.time() - beg) * 1000.0  # toc (ms)
        logger.debug("Duration of computing gradients on backpropagation: %8.4fms.", end)
        return cost, (nabla_w, nabla_b)

    def cost_single(self, features, label):
        """
        Costo total de la red neuronal para una entrada singular {*features*, *label*}, dado por:

        :math:`C(W, b; x, y) = C_{FP}(W, b; x, y) + L1 + L2`

        donde :math:`C_{FP}` es el costo obtenido al final del Forward Pass durante el algoritmo de Backpropagation,
        y los términos *L1* y *L2* corresponden a las normas de regularización calculadas con las funciones
        :func:`~learninspy.core.model.NeuralNetwork.l1` y  :func:`~learninspy.core.model.NeuralNetwork.l2`
        respectivamente.

        :param features: *numpy.ndarray*
        :param label: float o *numpy.ndarray*
        :return: float (costo), tuple de :class:`~learninspy.core.neurons.LocalNeurons` (gradientes de W y b)

        .. note:: Para problemas de clasificación, el float *label* se convierte a un vector binario
          de dimensión K (dado por la cantidad de clases a predecir) mediante
          :func:`~learninspy.utils.data.label_to_vector` para así poder aplicar una función de costo
          en forma directa contra la predicción realizada por la softmax (que es un vector).
        """
        # 'label' se debe vectorizar para que se pueda utilizar en una fun_loss sobre un Neurons (arreglo)
        # - Si la tarea de la red es de clasificación, entonces 'label' se debe convertir a vector binario.
        #   Así, se puede aplicar directamente a una función de costo contra la salida del softmax (vector).
        # - Si la tarea de la red es de regresión, entonces 'label' pasa a ser un list(label)
        if self.params.classification is True:
            label = label_to_vector(label, self.params.units_layers[-1])  # n_classes dado por la dim de la últ capa
        else:
            label = [label]  # Conversion a list necesaria para loss que opera con arreglos
        cost, (nabla_w, nabla_b) = self._backprop(features, label)

        if self.params.strength_l1 > 0.0:
            c, n_w = self.l1()
            cost += c
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, n_w))
        if self.params.strength_l2 > 0.0:
            c, n_w = self.l2()
            cost += c
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, n_w))

        return cost, (nabla_w, nabla_b)

    def cost_overall(self, data):
        """
        Costo promedio total de la red neuronal para un batch de M entradas {*features*, *label*}, dado por:

        :math:`C(W, b; x, y) = \\dfrac{1}{M} \displaystyle\sum\limits_{i}^M C_{FP}(W, b; x^{(i)}, y^{(i)}) + L1 + L2`

        donde :math:`C_{FP}` es el costo obtenido al final del Forward Pass durante el algoritmo de Backpropagation,
        y los términos *L1* y *L2* corresponden a las normas de regularización calculadas con las funciones
        :func:`~learninspy.core.model.NeuralNetwork.l1` y  :func:`~learninspy.core.model.NeuralNetwork.l2`
        respectivamente.

        :param data: list de *pyspark.mllib.regression.LabeledPoint*
        :return: float (costo), tuple de :class:`~learninspy.core.neurons.LocalNeurons` (gradientes de W y b)

        .. note:: Para problemas de clasificación, el float *label* se convierte a un vector binario
          de dimensión K (dado por la cantidad de clases a predecir) mediante
          :func:`~learninspy.utils.data.label_to_vector` para así poder aplicar una función de costo
          en forma directa contra la predicción realizada por la softmax (que es un vector).
        """
        # TODO: tener en cuenta esto http://stats.stackexchange.com/questions/108381/how-to-avoid-nan-in-using-relu-cross-entropy
        features = map(lambda lp: lp.features, data)
        if self.params.classification is True:
            labels = map(lambda lp: label_to_vector(lp.label, self.params.units_layers[-1]), data)
        else:
            labels = map(lambda lp: [lp.label], data)

        # Loss avg overall
        cost, (nabla_w, nabla_b) = self._backprop(features[0], labels[0])
        n = len(data)
        for f, l in zip(features, labels)[1:]:
            c, (n_w, n_b) = self._backprop(f, l)
            cost += c
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, n_w))
            nabla_b = map(lambda (n1, n2): n1 + n2, zip(nabla_b, n_b))
        cost /= float(n)
        nabla_w = map(lambda n_l: n_l / float(n), nabla_w)
        nabla_b = map(lambda n_l: n_l / float(n), nabla_b)

        # Regularization
        if self.params.strength_l1 > 0.0:
            c, n_w = self.l1()
            cost += c
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, n_w))
        if self.params.strength_l2 > 0.0:
            c, n_w = self.l2()
            cost += c
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, n_w))

        return cost, (nabla_w, nabla_b)

    def predict(self, x):
        """
        Predicciones sobre una entrada de datos (singular o conjunto).

        :param x: *numpy.ndarray* o *pyspark.mllib.regression.LabeledPoint*, o list de ellos.
        :return: *numpy.ndarray*
        """
        beg = time.time()  # tic
        if isinstance(x, list):
            x = map(lambda x_i: self.predict(x_i).matrix, x)
        else:
            if isinstance(x, LabeledPoint):
                x = x.features
            # Tener en cuenta que en la prediccion no se aplica el dropout
            for i in xrange(self.num_layers):
                x = self.list_layers[i].output(x, grad=False)
        end = (time.time() - beg) * 1000.0  # toc (ms)
        logger.debug("Duration of computing predictions to produce output : %8.4fms.", end)
        return x

    def check_stop(self, epochs, criterions, check_all=False):
        """
        Chequeo de los criterios de cortes definidos sobre la información del ajuste en una red neuronal.

        :param epochs: int, número de épocas efectuadas en el ajuste de la red
        :param criterions: list de *criterion*, instanciados desde :mod:`~learninspy.core.stops`.
        :param check_all: bool, si es True se devuelve un AND de todos los criterios y si es False
         se utiliza un OR.
        :return: bool, indicando True si los criterios señalan que se debe frenar el ajuste de la red.
        """
        if len(self.hits_valid) == 0:
            hits = 0.0
        else:
            hits = self.hits_valid[-1]
        results = {'hits': hits,
                   'iterations': epochs}
        if check_all is True:
            stop = all(c(results) for c in criterions)
        else:
            stop = any(c(results) for c in criterions)
        return stop

    def evaluate(self, data, predictions=False, measure=None):
        """
        Evaluación de un conjunto de datos etiquetados, midiendo la salida real o de predicción
        contra la esperada mediante una métrica definida en base a la tarea asignada para la red.

        :param data: instancia de :class:`.LabeledDataSet` o list de *pyspark.mllib.regression.LabeledPoint*
        :param predictions: bool, si es True se deben retornar también las predicciones hechas sobre *data*
        :param measure: string, key de alguna medida implementada en alguna de las métricas
         diponibles en :mod:`~learninspy.utils.evaluation`.
        :return: float, resultado de aplicar la medida dada por *measure*. Si *predictions* es True se retorna
         además una lista de *numpy.ndarray* (predicciones).
        """
        if isinstance(data, LabeledDataSet):
            actual = data.labels
            if type(data) is DistributedLabeledDataSet:
                actual = actual.collect()
                predicted = data.features.map(lambda f: self.predict(f).matrix).collect()
            else:
                predicted = map(lambda f: self.predict(f).matrix, data.features)
        else:
            actual = map(lambda lp: lp.label, data)
            predicted = map(lambda lp: self.predict(lp.features).matrix, data)
        if self.params.classification is True:
            # En problemas de clasificacion, se determina la prediccion por la unidad de softmax que predomina
            predicted = map(lambda p: float(np.argmax(p)), predicted)
            n_classes = self.params.units_layers[-1]  # La cant de unidades de la ult capa define la cant de clases
            metrics = ClassificationMetrics(zip(predicted, actual), n_classes=n_classes)
            if measure is None:
                measure = 'F-measure'
        else:
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

    def _train(self, train_bc, mini_batch=50, parallelism=4, measure=None, optimizer_params=None,
               reproducible=False, evaluate=True, seeds=None):
        """
        Entrenamiento de la red neuronal sobre el conjunto *train_bc*.

        :param train_bc: *pyspark.Broadcast*, variable Broadcast de Spark correspondiente al conjunto de entrenamiento.
        :param mini_batch: int, cantidad de ejemplos a utilizar durante una época de la optimización.
        :param parallelism: int, cantidad de modelos a optimizar concurrentemente.
            Si es -1, se setea como :math:`\\frac{N}{m}` donde *N* es la cantidad
            total de ejemplos de entrenamiento y *m* la cantidad de ejemplos para el mini-batch.
        :param optimizer_params: :class:`.OptimizerParameters`
        :param measure: string, key de alguna medida implementada en alguna de las métricas
         diponibles en :mod:`~learninspy.utils.evaluation`.
        :param reproducible: bool, si es True se indica que se debe poder reproducir exactamente el ajuste.
        :param evaluate: bool, si es True se evalua el modelo sobre el conjunto de entrenamiento.
        :return: float, resultado de evaluar el modelo final, o None si *evaluate* es False.
        """
        if parallelism == -1:
            # Se debe entrenar un modelo por cada batch (aunque se pueden solapar)
            total = len(train_bc.value)
            parallelism = total / mini_batch

        # Funcion para minimizar funcion de costo sobre cada modelo del RDD
        minimizer = opt.optimize
        # TODO ver si usar sc.accumulator para acumular actualizaciones y despues aplicarlas (paper de mosharaf)

        if reproducible is True:
            self._set_initial_rndstate()  # Seteo estado inicial del RandomState (al generarse la instancia NeuralNetwork)

        if seeds is None:
            seeds = list(self.params.rng.randint(500, size=parallelism))

        # Paralelizo modelo actual en los nodos mediante parallelism (que define el n° de particiones o slices del RDD)
        # NOTA: se persiste en memoria serializando ya que se supone que son objetos grandes y no conviene cachearlos
        models_rdd = (sc.parallelize(zip([self] * parallelism, seeds), numSlices=parallelism)
                        .persist(StorageLevel.MEMORY_ONLY_SER)  # TODO: si o si este StorageLevel?
                      )
        # Minimizo el costo de las redes en paralelo
        # NOTA: persist() es importante porque se traza varias veces el grafo de acciones sobre el RDD results
        logger.debug("Training %i models in parallel.", parallelism)
        results = (models_rdd.map(lambda (model, seed):
                                  minimizer(model, train_bc.value, optimizer_params, mini_batch, seed))
                             .persist(StorageLevel.MEMORY_ONLY_SER)  # TODO: si o si este StorageLevel?
                   )
        # Junto modelos entrenados en paralelo, en base a un criterio de ponderacion sobre un valor objetivo
        logger.debug("Merging models ...")
        if self.params.classification is True:
            list_layers = opt.merge_models(results, optimizer_params.merge['criter'], optimizer_params.merge['goal'])
        else:
            # Se realiza un promedio de hits sin ponderacion TODO cambiar esta distincion
            list_layers = opt.merge_models(results, criter='avg', goal='hits')
        # Copio el resultado de las capas mezcladas en el modelo actual
        self.list_layers = copy.copy(list_layers)
        # Quito de cache
        logger.debug("Unpersisting replicated models ...")
        results.unpersist()
        models_rdd.unpersist()
        if evaluate is True:
            # Evaluo tasa de aciertos de entrenamiento
            hits = self.evaluate(train_bc.value, predictions=False, measure=measure)
        else:
            hits = None
        return hits

    # TODO: crear un fit_params que abarque stops, parallelism, reproducible, keep_best, valid_iters y measure
    def fit(self, train, valid=None, mini_batch=50, parallelism=4, valid_iters=10, measure=None,
            stops=None, optimizer_params=None, reproducible=False, keep_best=False):
        """
        Ajuste de la red neuronal utilizando los conjuntos *train* y *valid*, mediante las siguientes pautas:

        * Durante la optimización, un modelo itera sobre un batch o subconjunto muestreado de *train* cuya
          magnitud está dada por *mini_batch*.
        * La optimización se realiza en forma distribuida, seleccionando batchs de datos para cada modelo
          a entrenar en forma paralela por cada iteración de la optimización. La cantidad de modelos a entrenar
          en forma concurrente está dada por *parallelism*.
        * El conjunto *train* es enviado en Broadcast de Spark a todos los nodos de ejecución para mejorar el esquema
          de comunicación durante la optimización.
        * La validación se realiza cada una cierta cantidad de épocas, dada por *valid_iters*,
          para así poder agilizar el ajuste cuando *valid* es de gran dimensión.
        * La optimización es controlada mediante los parámetros :class:`.OptimizerParameters` y los criterios de corte
          provenientes de :mod:`~learninspy.core.stops`.


        :param train: :class:`.LabeledDataSet` or list, conjunto de datos de entrenamiento.
        :param valid: :class:`.LabeledDataSet` or list, conjunto de datos de validación.
        :param mini_batch: int, cantidad de ejemplos a utilizar durante una época de la optimización.
        :param parallelism: int, cantidad de modelos a optimizar concurrentemente.
            Si es -1, se setea como :math:`\\frac{N}{m}` donde *N* es la cantidad
            total de ejemplos de entrenamiento y *m* la cantidad de ejemplos para el mini-batch.
        :param valid_iters: int, indicando cada cuántas iteraciones evaluar el modelo sobre el conjunto *valid*.
        :param measure: string, key de alguna medida implementada en alguna de las métricas
         diponibles en :mod:`~learninspy.utils.evaluation`.
        :param stops: list de *criterion*, instanciados desde :mod:`~learninspy.core.stops`.
        :param optimizer_params: :class:`.OptimizerParameters`
        :param reproducible: bool, si es True se indica que se debe poder reproducir exactamente el ajuste.
        :param keep_best: bool, indicando **True** si se desea mantener al final de la optimización
            el mejor modelo obtenido.
        :return: float, resultado de evaluar el modelo final sobre el conjunto *valid*.
        """
        # Si es LabeledDataSet, lo colecto en forma de lista
        if isinstance(train, LabeledDataSet):
            train = train.collect()
        # Creo Broadcasts, de manera de mandarlo una sola vez a todos los nodos
        logger.debug("Broadcasting training dataset ...")
        train = sc.broadcast(train)

        # TODO: ver si hay mas opciones que quedarse con el mejor
        if keep_best is True:
            best_epoch = 1
            best_valid = 0.0
            best_model = None

        if stops is None:
            logger.debug("Setting up stop criterions by default.")
            stops = [criterion['MaxIterations'](5),
                     criterion['AchieveTolerance'](0.95, key='hits')]

        if parallelism == -1:
            # Se debe entrenar un modelo por cada batch (aunque se pueden solapar)
            total = len(train.value)
            parallelism = total / mini_batch

        if reproducible is True:
            self._set_initial_rndstate()  # Seteo estado inicial del RandomState (al generarse la instancia NeuralNetwork)

        seeds = list(self.params.rng.randint(500, size=parallelism))

        # Inicializo variable a utilizar
        epoch = 0
        total_end = 0
        hits_train = 0.0
        # Reset de historial de train y valid (es preferible para ser consistente a lo largo de muchos fits)
        self.hits_valid = []
        self.hits_train = []
        self.epochs = []
        while self.check_stop(epoch, stops) is False:
            beg = time.time()  # tic
            evaluate = epoch % valid_iters == 0
            # De forma que los samples de datos varien en cada iteración, se cambian las seeds de forma determinística
            seeds = [s + epoch for s in seeds]

            hits_train = self._train(train, mini_batch=mini_batch, parallelism=parallelism, measure=measure,
                                     optimizer_params=optimizer_params, reproducible=reproducible,
                                     evaluate=evaluate, seeds=seeds)
            end = time.time() - beg  # toc
            total_end += end  # Acumular total
            # Validacion cada ciertas iteraciones, dado por valid_iters
            if evaluate is True:
                self.hits_train.append(hits_train)
                self.hits_valid.append(self.evaluate(valid, predictions=False, measure=measure))
                self.epochs.append(epoch + 1)
                logger.info("Epoca %i realizada en %8.4fs. Hits en train: %12.11f. Hits en valid: %12.11f",
                            epoch+1, end, self.hits_train[-1], self.hits_valid[-1])
                if keep_best is True:
                    if self.hits_valid[-1] >= best_valid:
                        best_epoch = epoch + 1
                        best_valid = self.hits_valid[-1]
                        best_model = self.list_layers
            else:
                logger.info("Epoca %i realizada en %8.4fs.",
                            epoch+1, end)
            # Recoleccion de basura manual para borrar de memoria los objetos largos generados por los datasets
            # Ver http://www.digi.com/wiki/developer/index.php/Python_Garbage_Collection
            collected = gc.collect()
            logger.debug("Garbage Collector: Recolectados %d objetos.", collected)
            epoch += 1
        if keep_best is True:
            if best_model is not None: # Si hubo algun best model, procedo con el reemplazo
                self.list_layers = copy.deepcopy(best_model)
                # Se truncan las siguientes listas hasta el indice correspondiente al best model
                i = self.epochs.index(best_epoch)
                self.epochs = self.epochs[:i]
                self.hits_train = self.hits_train[:i]
                self.hits_valid = self.hits_valid[:i]
        # Evaluación final  # TODO: vale la pena hacerla?
        self.epochs.append(epoch + 1)
        self.hits_train.append(self.evaluate(train.value, predictions=False, measure=measure))
        self.hits_valid.append(self.evaluate(valid, predictions=False, measure=measure))
        logger.info("Ajuste total realizado en %8.4fs. Hits en train: %12.11f. Hits en valid: %12.11f",
                    total_end, self.hits_train[-1], self.hits_valid[-1])
        logger.debug("Unpersisting training dataset...")
        train.unpersist()
        return self.hits_valid[-1]

    def update(self, step_w, step_b):
        """
        Actualiza los pesos sinápticos de cada capa en la red, agregando a cada una los incrementos
        ingresados como parámetros mediante llamadas a la función :func:`~learninspy.core.model.NeuralLayer.update`.

        :param step_w: list de :class:`~learninspy.core.neurons.LocalNeurons`.
        :param step_b: list de :class:`~learninspy.core.neurons.LocalNeurons`.
        :return: :class:`~learninspy.core.model.NeuralNetwork`.
        """
        # Cada parametro step es una lista, para actualizar cada capa
        for l in xrange(self.num_layers):
            self.list_layers[l].update(step_w[l], step_b[l])
        return self

    def save(self, filename):
        """
        Guardar el modelo en forma serializada con Pickler.

        :param filename: string, ruta indicando dónde debe almacenarse.

        >>> # Load
        >>> model_path = '/tmp/model/test_model.lea'
        >>> test_model = NeuralNetwork.load(filename=model_path)
        >>> # Save
        >>> test_model.save(filename=model_path)
        """
        f = open(filename, 'w')
        pickler = pickle.Pickler(f, -1)
        pickler.dump(self)
        f.close()
        return

    @classmethod
    def load(cls, filename):
        """
        Carga de un modelo desde archivo en forma serializada con Pickler.

        :param filename: string, ruta indicando desde dónde debe cargarse.
        :return: :class:`~learninspy.core.model.NeuralNetwork`
        """
        f = open(filename)
        model = pickle.load(f)
        f.close()
        return model

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
