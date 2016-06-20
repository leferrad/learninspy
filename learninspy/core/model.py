#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np
from scipy import sparse

# Dependencias internas
from learninspy.core import activations as act, loss, optimization as opt
from learninspy.core.stops import criterion
from learninspy.core.neurons import LocalNeurons
from learninspy.utils import checks
from learninspy.utils.evaluation import ClassificationMetrics, RegressionMetrics
from learninspy.utils.data import LabeledDataSet
from learninspy.context import sc
from learninspy.utils.fileio import get_logger

# Librerias de Python
import copy
import cPickle as pickle
import os
import time

logger = get_logger(name=__name__)
logger.propagate = False  # Para que no se dupliquen los mensajes por herencia


class NeuralLayer(object):
    """
    Clase básica para modelar una capa de neuronas que compone una red neuronal.
    Contiene sus "neuronas" representadas por pesos sinápticos **w** y **b**,
    además de una función de activación asociada para dichos pesos.

    Una correcta inicialización de los pesos sinápticos está muy ligada a la función de activación elegida.
    Por defecto, los pesos sinápticos se inicializan con una distribución uniforme
    con media :math:`0` y varianza :math:`\\frac{2.0}{\sqrt{n_{in}}}`,
    lo cual da buenos resultados especialmente usando ReLUs.
    Para la función *Tanh* se muestrea sobre una distribución uniforme
    en el rango :math:`\pm \sqrt{\\frac{6}{n_{in}+n_{out}}}`, y para la *Sigmoid* en el rango
    :math:`\pm 4.0 \sqrt{\\frac{6}{n_{in}+n_{out}}}`.

    :param n_in: int, dimensión de la entrada.
    :param n_out: int, dimensión de la salida.
    :param activation: string, key de la función de activación asignada a la capa.
    :param distribute: si es True, indica que se utilicen arreglos distribuidos para **w** y **b**.
    :param w: :class:`.LocalNeurons`, matriz de pesos sinápticos.
    :param b: :class:`.LocalNeurons`, vector de pesos bias.
    :param sparsity: si es True, los arreglos se almacenan en formato **scipy.sparse.csr_matrix**.
    :param rng: si es *None*, se crea un generador de números aleatorios mediante una instancia **numpy.random.RandomState**.

    .. note:: el parámetro *distribute* no tiene efecto, ya que el uso de arreglos distribuidos se deja para un próximo release.

    """

    def __init__(self, n_in=2, n_out=2, activation='ReLU', distributed=False, w=None, b=None, sparsity=False, rng=None):
        self.n_out = n_out
        self.n_in = n_in
        self.activation = act.fun_activation[activation]
        self.activation_d = act.fun_activation_d[activation]
        sparsity = False  # TODO completar esta funcionalidad
        distributed = False

        if rng is None:
            rng = np.random.RandomState(123)
        self.rng = rng
        self.rnd_state = self.rng.get_state()

        self.shape_w = n_out, n_in
        self.shape_b = n_out, 1

        #  Recomendaciones de http://cs231n.github.io/neural-networks-2/ y http://deeplearning.net/tutorial/mlp.html#mlp
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
            if sparsity is True:
                w = sparse.csr_matrix(w)  # Convierto matriz densa a "rala"

        if b is None:
            b = np.zeros(self.shape_b, dtype=np.dtype(float))

            if sparsity is True:  # Hago que el vector sea sparse
                b = sparse.csr_matrix(b)

        # TODO weights_T era p/ poder hacer operaciones distribuidas, pero se deja como experimental DistributedNeurons
        if distributed is True:
            logger.error("DistributedNeurons will be implemented soon ...")
            #self.weights = DistributedNeurons(w, self.shape_w)
            #self.weights_T = DistributedNeurons(w.T, self.shape_w[::-1])
            #self.bias = DistributedNeurons(b, self.shape_b)
        else:
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
        Norma **L1** sobre la matriz **w** de pesos sinápticos.
        :return: float, resultado de aplicar norma.
        """
        return self.weights.l1()

    def l2(self):
        """
        Norma **L2** sobre la matriz **w** de pesos sinápticos.
        :return: float, resultado de aplicar norma.
        """
        return self.weights.l2()

    def output(self, x, grad=False):
        """
        Salida de la capa neuronal. Se toma una entrada :math:`x \in \Re^{n_{in}}`, se pondera con los
        pesos sinápticos **W** y el bias **b**, y luego se aplica la función de activación **f** para retornar el
        resultado :math:`a = f(Wx + b)`.

        :param x: **numpy.array**, vector de entrada
        :param grad: Si es *True*, se retorna además el gradiente de la salida.
        :return: **numpy.array**, o tupla de ellos si *grad* es True.
        """
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.activation(self.activation)
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)
        return a

    # Basado en http://cs231n.github.io/neural-networks-2/
    def dropoutput(self, x, p, grad=False):
        """
        Salida de la capa neuronal, luego de aplicar la regularización de los pesos sinápticos por Dropout.


        http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

        :param x: **numpy.array**, vector de entrada
        :param p: float, tal que :math:`0<p<1`
        :param grad: Si es *True*, se retorna además el gradiente de la salida.
        :return: **numpy.array**, o tupla de ellos si *grad* es True.

        .. note:: En las predicciones de la red no se debe efectuar Dropout.
        """
        self.rng.set_state(self.rnd_state)  # Para que sea reproducible
        out = self.output(x, grad)
        if grad:
            a, d_a = out
            a, mask = a.dropout(p, self.rng.randint(500))
            out = a, d_a
        else:
            out, mask = out.dropout(p, self.rng.randint(500))
        return out, mask  # Devuelvo ademas la mascara utilizada, para el backprop

    def get_weights(self):
        """
        Se devuelve la matriz de pesos sinápticos **w**.

        :return: **numpy.array**.
        """
        return self.weights

    def get_bias(self):
        """
        Se devuelve el vector de bias **b**.

        :return: **numpy.array**.
        """
        return self.bias

    def _persist_layer(self):
        self.weights._persist()
        self.bias._persist()
        return

    def _unpersist_layer(self):
        self.weights._unpersist()
        self.bias._unpersist()
        return

    def update(self, step_w, step_b):  # Actualiza sumando los argumentos w y b a los respectivos pesos
        """
        Se actualizan los arreglos **w** y **b** sumando respectivamente los incrementos
        recibidos por parámetros.

        :param step_w: :class:`.LocalNeurons`
        :param step_b: :class:`.LocalNeurons`
        """
        self.weights += step_w
#        self.weights_T += step_w.transpose()
        self.bias += step_b
        return


class ClassificationLayer(NeuralLayer):
    """
    Clase correspondiente a la capa de salida en una red neuronal con tareas de clasificación.
    Se distingue de una :class:`.RegressionLayer` en que para realizar la clasificación se define
    que la activación se de por la función *softmax*.

    """

    def output(self, x, grad=False):
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.softmax()  # La activacion es un clasificador softmax
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)
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


# TODO: Ver la factibilidad de cambiarlo por un dict
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
    """
    def __init__(self, units_layers, activation='ReLU', layer_distributed=None, dropout_ratios=None,
                 classification=True, strength_l1=1e-5, strength_l2=1e-4, seed=123):
        num_layers = len(units_layers)  # Cant total de capas (entrada + ocultas + salida)
        if dropout_ratios is None:
            if classification is True:
                # Default, dropout de 0.2 en entrada, de 0.5 en ocultas y la salida no debe tener
                dropout_ratios = [0.2] + [0.5] * (num_layers-2) + [0.0]
            else:
                dropout_ratios = [0.0] * num_layers  # Nunca recomendado hacer dropout en regresion
            dropout_ratios[-1] = 0.0  # TODO es para asegurarme que no haya DropOut en la salida, pero verificar mejor
        if layer_distributed is None:
            layer_distributed = [False] * num_layers   # Por defecto, las capas no estan distribuidas
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

    def __str__(self):
        config = ""
        for l in xrange(len(self.units_layers)):
            if l == (len(self.units_layers)-1) and self.classification is True:  # Para especificar softmax de clasific
                config += "Layer "+str(l)+" with "+str(self.units_layers[l])+" neurons, using " \
                          + "Softmax activation."+os.linesep
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
    :param list_layers: list of :class:`.NruralLayer`, en caso de que se utilicen capas de neuronas
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
        self.hits_train = 0.0
        self.hits_valid = 0.0
        #self.check_gradients()  # Validar los gradientes de las funciones de activacion y error elegidas
        #self.check_gradients()  # Validar los gradientes de las funciones de activacion y error elegidas

    def __assert_type_outputlayer(self):
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
        self.set_initial_rndstate()
        for i in xrange(1, num_layers - 1):
            self.list_layers.append(NeuralLayer(self.params.units_layers[i - 1], self.params.units_layers[i],
                                                self.params.activation[i - 1], self.params.layer_distributed[i],
                                                rng=self.params.rng))
        if self.params.classification is True:
            # Ultima capa es de clasificacion, por lo que su activacion es softmax
            self.list_layers.append(ClassificationLayer(self.params.units_layers[-2],
                                                        self.params.units_layers[-1],
                                                        self.params.activation[-2],
                                                        # Notar que no tomo la ultima, ya que la lista tiene 1 elem mas
                                                        self.params.layer_distributed[-1],
                                                        rng=self.params.rng))
        else:
            # Ultima capa es de regresion, por lo que su activacion puede ser cualquiera
            # (la misma que las ocultas por default)
            self.list_layers.append(RegressionLayer(self.params.units_layers[num_layers - 2],
                                                    self.params.units_layers[num_layers - 1],
                                                    self.params.activation[-1],
                                                    # Notar que ahora tomo la ultima, ya que se corresponde a la salida
                                                    self.params.layer_distributed[num_layers - 1],
                                                    rng=self.params.rng))

    def l1(self):
        """
        Norma **L1** sobre todas las capas (explicar que es una suma y luego se multiplica por strength).
        :return: float, resultado de aplicar norma.
        """
        cost = 0.0
        gradient = []
        for layer in self.list_layers:
            c_l1, g_l1 = layer.l1()
            cost += c_l1
            gradient.append(g_l1)
        cost *= self.params.strength_l1
        gradient[:] = [grad * self.params.strength_l1 for grad in gradient]
        return cost, gradient

    # Ante la duda, l2 consigue por lo general mejores resultados que l1 pero se pueden combinar
    def l2(self):
        cost = 0.0
        gradient = []
        for layer in self.list_layers:
            c_l2, g_l2 = layer.l2()
            cost += c_l2
            gradient.append(g_l2)
        cost *= self.params.strength_l2 * 0.5  # Multiplico por 0.5 para hacer mas simple el gradiente
        gradient[:] = [grad * self.params.strength_l2 for grad in gradient]
        return cost, gradient

    def set_l1(self, strength_l1):
        self.params.strength_l1 = strength_l1
        return

    def set_l2(self, strength_l2):
        self.params.strength_l2 = strength_l2
        return

    def set_dropout_ratios(self, dropout_ratios):
        self.params.dropout_ratios = dropout_ratios
        return

    def _backprop(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        # Ver http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
        # x, y: numpy array
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

    def cost(self, features, label):
        cost, (nabla_w, nabla_b) = self._backprop(features, label)
        if self.params.strength_l1 > 0.0:
            cost_l1, nabla_w_l1 = self.l1()
            cost += cost_l1
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, nabla_w_l1))
        if self.params.strength_l2 > 0.0:
            cost_l2, nabla_w_l2 = self.l2()
            cost += cost_l2
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, nabla_w_l2))
        return cost, (nabla_w, nabla_b)

    def predict(self, x):
        beg = time.time()  # tic
        if isinstance(x, list):
            x = map(lambda lp: self.predict(lp.features).matrix(), x)
        else:
            # Tener en cuenta que en la prediccion no se aplica el dropout
            for i in xrange(self.num_layers):
                x = self.list_layers[i].output(x, grad=False)
        end = (time.time() - beg) * 1000.0  # toc (ms)
        logger.debug("Duration of computing predictions to produce output : %8.4fms.", end)
        return x

    def check_stop(self, epochs, criterions, check_all=False):
        results = {'hits': self.hits_valid,
                   'iterations': epochs}
        if check_all is True:
            stop = all(c(results) for c in criterions)
        else:
            stop = any(c(results) for c in criterions)
        return stop

    def evaluate(self, data, predictions=False, metric=None):
        """

        :param data:
        :param predictions: bool, for returning predictions too
        :return:
        """
        if type(data) is LabeledDataSet:
            data = data.collect()
        actual = map(lambda lp: lp.label, data)
        predicted = map(lambda lp: self.predict(lp.features).matrix, data)
        if self.params.classification is True:
            # En problemas de clasificacion, se determina la prediccion por la unidad de softmax que predomina
            predicted = map(lambda p: float(np.argmax(p)), predicted)
        if self.params.classification is True:
            n_classes = self.params.units_layers[-1]  # La cant de unidades de la ult capa define la cant de clases
            metrics = ClassificationMetrics(zip(predicted, actual), n_classes=n_classes)
            if metric is None:
                metric = 'F-measure'
            # TODO mejorar esto para que quede mas prolijo y generalizable
            if metric == 'Accuracy':
                hits = metrics.accuracy()
            elif metric == 'F-measure':
                hits = metrics.f_measure()
            elif metric == 'Precision':
                hits = metrics.precision()
            elif metric == 'Recall':
                hits = metrics.recall()
        else:
            metrics = RegressionMetrics(zip(predicted, actual))
            if metric is None:
                metric = 'R2'
            # TODO mejorar esto para que quede mas prolijo y generalizable
            if metric == 'R2':
                hits = metrics.r2()
            elif metric == 'MSE':
                hits = metrics.mse()
        if predictions is True:  # Devuelvo ademas el vector de predicciones
            ret = hits, predicted
        else:
            ret = hits
        return ret

    def train(self, train_bc, mini_batch=50, parallelism=4, optimizer_params=None, reproducible=False):
        """

        :param train_bc:
        :param mini_batch:
        :param parallelism:
        :param optimizer_params:
        :return:
        """
        if parallelism == -1:
            # Se debe entrenar un modelo por cada batch (aunque se pueden solapar)
            total = len(train_bc.value)
            parallelism = total / mini_batch

        # Funcion para minimizar funcion de costo sobre cada modelo del RDD
        minimizer = opt.optimize
        # TODO ver si usar sc.accumulator para acumular actualizaciones y despues aplicarlas (paper de mosharaf)

        if reproducible is True:
            self.set_initial_rndstate()  # Seteo estado inicial del RandomState (al generarse la instancia NeuralNetwork)

        seeds = list(self.params.rng.randint(500, size=parallelism))
        # Paralelizo modelo actual en los nodos dados por parallelism
        models_rdd = sc.parallelize(zip([self] * parallelism, seeds))
        # Minimizo el costo de las redes en paralelo
        # NOTA: cache() es importante porque se traza varias veces el grafo de acciones sobre el RDD results
        logger.debug("Training %i models in parallel.", parallelism)
        results = models_rdd.map(lambda (model, seed):
                                 minimizer(model, train_bc.value, optimizer_params, mini_batch, seed)).cache()
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
        results.unpersist()
        models_rdd.unpersist()
        # Evaluo tasa de aciertos de entrenamiento
        hits = self.evaluate(train_bc.value)
        return hits

    def fit(self, train, valid=None, stops=None, mini_batch=50, parallelism=4, optimizer_params=None,
            reproducible=False, keep_best=False):
        """

        :param train: :class:`.LabeledDataSet` or list, conjunto de datos de entrenamiento.
        :param valid: :class:`.LabeledDataSet` or list, conjunto de datos de validación.
        :param mini_batch: int, cantidad de ejemplos a utilizar durante una época de la optimización.
        :param parallelism: int, cantidad de modelos a optimizar concurrentemente.
            Si es -1, se setea como :math:`\\frac{N_{train}}{m}` donde *N* es la cantidad
            total de ejemplos de entrenamiento y *m* la cantidad de ejemplos para el mini-batch.
        :param stops: list of Criterions.
        :param optimizer_params: :class:`.OptimizerParameters`
        :param keep_best: bool, indicando **True** si se desea mantener al final de la optimización
            el mejor modelo obtenido.
        """
        # Si son LabeledDataSet, los colecto en forma de lista
        if type(train) is LabeledDataSet:
            train = train.collect()
        if type(valid) is LabeledDataSet:
            valid = valid.collect()
        # Creo Broadcasts, de manera de mandarlo una sola vez a todos los nodos
        logger.debug("Broadcasting datasets ...")
        train_bc = sc.broadcast(train)
        valid_bc = sc.broadcast(valid)  # Por ahora no es necesario el bc, pero puede q luego lo use en batchs p/ train
        # TODO: ver si hay mas opciones que quedarse con el mejor
        if keep_best is True:
            best_model = self.list_layers
            best_valid = 0.0
            best_train = 0.0
        if stops is None:
            logger.debug("Set up stop criterions by default.")
            stops = [criterion['MaxIterations'](5),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        epoch = 0
        total_end = 0
        while self.check_stop(epoch, stops) is False:
            beg = time.time()  # tic
            self.hits_train = self.train(train_bc, mini_batch, parallelism, optimizer_params, reproducible)
            self.hits_valid = self.evaluate(valid_bc.value)
            #print "Epoca ", epoch+1, ". Hits en train: ", self.hits_train, ". Hits en valid: ", self.hits_valid
            end = time.time() - beg  # toc
            total_end += end  # Acumular total
            logger.info("Epoca %i realizada en %8.4fs. Hits en train: %12.11f. Hits en valid: %12.11f",
                        epoch+1, end, self.hits_train, self.hits_valid)
            if keep_best is True:
                if self.hits_valid >= best_valid:
                    best_valid = self.hits_valid
                    best_train = self.hits_train
                    best_model = self.list_layers
            epoch += 1
        if keep_best is True:
            self.list_layers = copy.deepcopy(best_model)
            self.hits_train = best_train
            self.hits_valid = best_valid
        logger.info("Ajuste total realizado en %8.4fs. Hits en train: %12.11f. Hits en valid: %12.11f",
                    total_end, self.hits_train, self.hits_valid)
        return self.hits_valid

    def update(self, stepw, stepb):
        # Cada parametro step es una lista, para actualizar cada capa
        for l in xrange(self.num_layers):
            self.list_layers[l].update(stepw[l], stepb[l])

    def check_gradients(self):
        fun_act = self.params.activation
        fun_loss = self.params.loss
        if type(fun_act) is not list:
            fun_act = [fun_act]
        # Chequeo funciones de activacion
        check = checks.CheckGradientActivation(fun_act)
        bad_gradients = check()
        if bad_gradients is None:
            print 'Gradientes de activaciones OK!'
        else:
            indexes = np.array(range(self.num_layers))
            index_badgrad = indexes[bad_gradients]
            raise Exception('El gradiente de las capas ' + str(index_badgrad) + ' se encuentra mal implementado!')
        # Chequeo funcion de error
        #
        # check = checks.CheckGradientLoss(fun_loss)
        # bad_gradient = check()
        # if bad_gradient is None:
        #     print 'Gradiente de loss OK!'
        # else:
        #     raise Exception('El gradiente de las funcion de error se encuentra mal implementado!')

    def set_initial_rndstate(self):
        self.params.rng.set_state(self.rnd_state)
        return

    def persist_layers(self):
        logger.warning("Persisting RDDs from NeuralLayer objects")
        for i in xrange(len(self.list_layers)):
            self.list_layers[i].persist_layer()

    def unpersist_layers(self):
        logger.warning("Unpersisting RDDs from NeuralLayer objects")
        for i in xrange(len(self.list_layers)):
            self.list_layers[i].unpersist_layer()

    def save(self, name, path):
        file = open(path+name+'.lea', 'w')
        pickler = pickle.Pickler(file, -1)
        pickler.dump(self)
        file.close()
        return

    def load(self, name, path):
        file = open(path+name+'.lea')
        model = pickle.load(file)
        self.__dict__.update(model.__dict__)
        return

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
