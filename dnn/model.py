__author__ = 'leferrad'

# Modulos a exportar
__all__ = ['NeuralLayer', 'DeepLearningParams', 'NeuralNetwork']

import numpy as np
import activations as act
import optimization as opt
import loss
from neurons import DistributedNeurons, LocalNeurons
from scipy import sparse
from pyspark.mllib.regression import LabeledPoint
from context import sc
#from utils.util import LearninspyLogger
import itertools
import copy
import checks
import time
import utils.util as util

class NeuralLayer(object):
    activation = act.relu
    activation_d = act.relu_d

    def __init__(self, n_in=2, n_out=2, activation='ReLU', distribute=True, w=None, b=None, sparsity=False, rng=None):
        self.n_out = n_out
        self.n_in = n_in
        self.activation = act.fun_activation[activation]
        self.activation_d = act.fun_activation_d[activation]
        self.sparsity = sparsity

        if rng is None:
            rng = np.random.RandomState(1234)
        self.rng = rng
        self.shape_w = n_out, n_in
        self.shape_b = n_out, 1

        if w is None:
            w = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6.0 / (n_in + n_out)),
                    high=+np.sqrt(6.0 / (n_in + n_out)),
                    size=self.shape_w),
                dtype=np.dtype(float)
            )
            if sparsity is True:
                w = sparse.csr_matrix(w)  # Convierto matriz densa a "rala"

        if b is None:
            b = np.zeros(self.shape_b, dtype=np.dtype(float))

            if sparsity is True:  # Hago que el vector sea sparse
                b = sparse.csr_matrix(b)

        if distribute is True:
            self.weights = DistributedNeurons(w, self.shape_w)
            self.weights_T = DistributedNeurons(w.T, self.shape_w[::-1])
            self.bias = DistributedNeurons(b, self.shape_b)
        else:
            self.weights = LocalNeurons(w, self.shape_w)
            self.weights_T = LocalNeurons(w.transpose(), self.shape_w[::-1])
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
        return self.weights.l1()

    def l2(self):
        return self.weights.l2()

    def output(self, x, grad=False):
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.activation(self.activation)
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)
        return a

    # Basado en http://cs231n.github.io/neural-networks-2/

    def dropoutput(self, x, p, grad=False):
        out = self.output(x, grad)
        if grad:
            a, d_a = out
            a, mask = a.dropout(p)
            out = a, d_a
        else:
            out, mask = out.dropout(p)
        return out, mask  # Devuelvo ademas la mascara utilizada, para el backprop

    # NOTA: durante la etapa de testeo no se debe usar dropout

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def persist_layer(self):
        self.weights.persist()
        self.bias.persist()
        return

    def unpersist_layer(self):
        self.weights.unpersist()
        self.bias.unpersist()
        return

    def update(self, step_w, step_b):  # Actualiza sumando los argumentos w y b a los respectivos pesos
        self.weights += step_w
#        self.weights_T += step_w.transpose()
        self.bias += step_b
        return


class ClassificationLayer(NeuralLayer):
    def output(self, x, grad=False):
        wx = self.weights.mul_array(x)
        z = wx.sum_array(self.bias)
        a = z.softmax()  # La activacion es un clasificador softmax
        if grad:
            d_a = z.activation(self.activation_d)
            a = (a, d_a)
        return a

    def dropoutput(self, x, p, grad=False):
        raise Exception("Don't use dropout for output layer")


class RegressionLayer(NeuralLayer):
    #  El output es el mismo que una NeuralLayer
    def dropoutput(self, x, p, grad=False):
        raise Exception("Don't use dropout for output layer")

# TODO: Ver la factibilidad de cambiarlo por un dict
class DeepLearningParams:

    def __init__(self, units_layers, activation='ReLU', loss='MSE', layer_distributed=None, dropout_ratios=None,
                 minimizer='Adadelta', autoencoder=False, rng=None):
        if dropout_ratios is None:
            dropout_ratios = len(units_layers) * [0.3]  # Por defecto, todos los radios son de 0.3
        if layer_distributed is None:
            layer_distributed = len(units_layers) * [False]  # Por defecto, las capas no estan distribuidas
        self.dropout_ratios = dropout_ratios  # Recordar que la capa de salida no sufre dropout
        # 'units_layers' es un vector que indica la cantidad de unidades de cada capa (1er elemento se refiere a capa visible)
        self.units_layers = units_layers
        self.layer_distributed = layer_distributed
        self.activation = activation
        self.loss = loss
        self.minimizer = minimizer
        self.autoencoder = autoencoder
        self.strength_l1 = 1e-5
        self.strength_l2 = 1e-4
        if rng is None:
            rng = np.random.RandomState(1234)
        self.rng = rng


class NeuralNetwork(object):
    def __init__(self, params=None, list_layers=None):
        self.params = params
        self.list_layers = list_layers  # En caso de que la red reciba capas ya inicializadas
        self.loss = loss.fun_loss[self.params.loss]
        self.loss_d = loss.fun_loss_d[self.params.loss]
        self.opt_algo = opt.Minimizer[self.params.minimizer]
        #self.logger = LearninspyLogger()

        if list_layers is None:
            self.list_layers = []  # Creo un arreglo vacio para ir agregando las capas que se inicializan
            self.__init_weights()

        self.num_layers = len(self.list_layers)
        self.check_gradients()  # Validar los gradientes de las funciones de activacion y error elegidas

    def __init_weights(self):  # Metodo privado
        num_layers = len(self.params.units_layers)
        for i in xrange(1, num_layers - 1):
            self.list_layers.append(NeuralLayer(self.params.units_layers[i - 1], self.params.units_layers[i],
                                                self.params.activation, self.params.layer_distributed[i]))
        # Ultima capa es de clasificacion, por lo que su activacion es softmax
        self.list_layers.append(ClassificationLayer(self.params.units_layers[num_layers - 2],
                                            self.params.units_layers[num_layers - 1],
                                            self.params.activation,
                                            self.params.layer_distributed[num_layers - 1]))

    def l1(self):
        cost = 0.0
        gradient = []
        for layer in self.list_layers:
            c_l1, g_l1 = layer.l1()
            cost += c_l1
            gradient.append(g_l1)
        cost = cost * self.params.strength_l1
        gradient[:] = [grad * self.params.strength_l1 for grad in gradient]
        return cost, gradient


    def l2(self):
        cost = 0.0
        gradient = []
        for layer in self.list_layers:
            c_l2, g_l2 = layer.l2()
            cost += c_l2
            gradient.append(g_l2)
        cost = cost * self.params.strength_l2 * 0.5  # Multiplico por 0.5 para hacer mas simple el gradiente
        gradient[:] = [grad * self.params.strength_l2 for grad in gradient]
        return cost, gradient


    # Ante la duda, l2 consigue por lo general mejores resultados que l1 pero se pueden combinar

    def predict(self, x):
        num_layers = self.num_layers
        # Tener en cuenta que en la prediccion no se aplica el dropout
        for i in xrange(num_layers):
            x = self.list_layers[i].output(x, grad=False)
        return x

    def backprop(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        # Ver http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
        # x, y: numpy array
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
        # TODO: tener en cuenta que la ultima capa no es softmax (segun UFLDL)
        d_cost = a[-1].loss_d(self.loss_d, y)
        delta = d_cost.mul_elemwise(d_a[-1])
        if drop_fraction[-1] > 0.0:  # No actualizo las unidades "tiradas"
            delta = delta.mul_elemwise(mask[-1])
        nabla_w[-1] = delta.outer(a[-2])
        nabla_b[-1] = delta
        for l in xrange(2, num_layers + 1):
            w_t = self.list_layers[-l + 1].weights_T
            delta = w_t.mul_array(delta).mul_elemwise(d_a[-l])
            if drop_fraction[-l] > 0.0:  # No actualizo las unidades "tiradas"
                delta = delta.mul_elemwise(mask[-l])
            nabla_w[-l] = delta.outer(a[-l - 1])
            nabla_b[-l] = delta
        return cost, (nabla_w, nabla_b)

    def cost(self, features, label):
        cost, (nabla_w, nabla_b) = self.backprop(features, label)
        if self.params.strength_l1 > 0.0:
            cost_l1, nabla_w_l1 = self.l1()
            cost += cost_l1
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, nabla_w_l1))
        if self.params.strength_l2 > 0.0:
            cost_l2, nabla_w_l2 = self.l2()
            cost += cost_l2
            nabla_w = map(lambda (n1, n2): n1 + n2, zip(nabla_w, nabla_w_l2))
        return cost, (nabla_w, nabla_b)

    def evaluate(self, data):
        hits = 0.0
        for lp in data:  # Por cada LabeledPoint del conj de datos
            out = self.predict(lp.features).matrix()
            pred = np.argmax(out)
            if pred == lp.label:
                hits += 1.0
        if type(data) is itertools.chain:
            data = list(data)
        size = len(data)
        hits /= float(size)
        return hits

    def train(self, data, label, mini_batch=50, epochs=50, parallelism=10, options=None):
        # x, y son np.ndarray
        # TODO: ver improve_patience en deeplearning.net
        assert data.shape[0] == label.shape[0], 'Datos y labels deben tener igual cantidad de entradas'
        labeled_data = map(lambda (x, y): LabeledPoint(y, x), zip(data, label))
        part = parallelism
        data_bc = sc.broadcast(labeled_data)  # creo un Broadcast, de manera de mandarlo una sola vez a todos los nodos
        # Funciones a usar en los RDD
        minimizer = opt.optimize
        mixer = opt.mix_models
        for ep in xrange(epochs):
            print "Epoca ", ep
            # TODO ver si usar sc.accumulator para acumular actualizaciones y despues aplicarlas (paper de mosharaf)
            seeds = list(self.params.rng.randint(100, size=parallelism))
            models_rdd = sc.parallelize(zip([self] * parallelism, seeds))
            results = models_rdd.map(lambda (model, seed): minimizer(model, data_bc.value, mini_batch, options, seed)).cache()
            layers = (results.map(lambda res: [layer * res['hits'] for layer in res['model']])
                             .reduce(lambda left, right: mixer(left, right)))
            # Se realiza un promedio ponderado por la tasa de aciertos en el train
            total_hits = results.map(lambda res: res['hits']).sum()
            final_list_layers = map(lambda layer: layer / total_hits, layers)
            self.list_layers = copy.copy(final_list_layers)
        hits = self.evaluate(data_bc.value)
        return hits

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

    def persist_layers(self):
        for i in xrange(len(self.list_layers)):
            self.list_layers[i].persist_layer()

    def unpersist_layers(self):
        for i in xrange(len(self.list_layers)):
            self.list_layers[i].unpersist_layer()



