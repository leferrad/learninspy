__author__ = 'leferrad'

# Modulos a exportar
__all__ = ['NeuralLayer', 'DeepLearningParams', 'NeuralNetwork']

# Dependencias externas
import numpy as np
from scipy import sparse

# Librerias de Learninspy
import activations as act
import optimization as opt
import loss
from stops import criterion
from neurons import DistributedNeurons, LocalNeurons
from context import sc
from evaluation import ClassificationMetrics, RegressionMetrics
from utils.data import label_data
import utils.util as util
#from utils.util import LearninspyLogger

# Librerias de Python
import copy
import checks
import cPickle as pickle
import time

class NeuralLayer(object):

    def __init__(self, n_in=2, n_out=2, activation='ReLU', distribute=False, w=None, b=None, sparsity=False, rng=None):
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
    #  Importante leer 'Word of caution' en http://cs231n.github.io/neural-networks-2/
    #  El output es el mismo que una NeuralLayer
    def dropoutput(self, x, p, grad=False):
        raise Exception("Don't use dropout for output layer")

# TODO: Ver la factibilidad de cambiarlo por un dict
class DeepLearningParams:

    def __init__(self, units_layers, activation='ReLU', layer_distributed=None, dropout_ratios=None,
                 classification=True, strength_l1=1e-5, strength_l2=1e-4, seed=123):
        num_layers = len(units_layers)  # Cant total de capas (entrada + ocultas + salida)
        if dropout_ratios is None:
            if classification is True:
                dropout_ratios = [0.5] * (num_layers-1) + [0.0]  # Default, dropout de 0.5 menos la salida que no debe tener
            else:
                dropout_ratios = [0.0] * num_layers  # Nunca recomendado hacer dropout en regresion
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



class NeuralNetwork(object):
    def __init__(self, params=None, list_layers=None):
        if params is None:
            params = DeepLearningParams([3, 3, 3])  # Creo cualquier cosa por defecto, para que no explote TODO: cambiar!
        self.params = params
        self.list_layers = list_layers  # En caso de que la red reciba capas ya inicializadas
        self.loss = loss.fun_loss[self.params.loss]
        self.loss_d = loss.fun_loss_d[self.params.loss]
        #self.logger = LearninspyLogger()

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
        num_layers = len(self.params.units_layers)
        for i in xrange(1, num_layers - 1):
            self.list_layers.append(NeuralLayer(self.params.units_layers[i - 1], self.params.units_layers[i],
                                                self.params.activation[i - 1], self.params.layer_distributed[i]))
        if self.params.classification is True:
            # Ultima capa es de clasificacion, por lo que su activacion es softmax
            self.list_layers.append(ClassificationLayer(self.params.units_layers[-2],
                                                        self.params.units_layers[-1],
                                                        self.params.activation[-2],
                                                        # Notar que no tomo la ultima, ya que la lista tiene 1 elem mas
                                                        self.params.layer_distributed[-1]))
        else:
            # Ultima capa es de regresion, por lo que su activacion puede ser cualquiera
            # (la misma que las ocultas por default)
            self.list_layers.append(RegressionLayer(self.params.units_layers[num_layers - 2],
                                                    self.params.units_layers[num_layers - 1],
                                                    self.params.activation[-1],
                                                    # Notar que ahora tomo la ultima, ya que se corresponde a la salida
                                                    self.params.layer_distributed[num_layers - 1]))

    def l1(self):
        cost = 0.0
        gradient = []
        for layer in self.list_layers:
            c_l1, g_l1 = layer.l1()
            cost += c_l1
            gradient.append(g_l1)
        cost *= self.params.strength_l1
        gradient[:] = [grad * self.params.strength_l1 for grad in gradient]
        return cost, gradient


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
    # Ante la duda, l2 consigue por lo general mejores resultados que l1 pero se pueden combinar

    def _backprop(self, x, y):
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
        if isinstance(x, list):
            x = map(lambda lp: self.predict(lp.features).matrix(), x)
        else:
            # Tener en cuenta que en la prediccion no se aplica el dropout
            for i in xrange(self.num_layers):
                x = self.list_layers[i].output(x, grad=False)
        return x

    def check_stop(self, epochs, criterions, check_all=False):
        results = {'hits': self.hits_valid,
                   'iterations': epochs}
        if check_all is True:
            stop = all(c(results) for c in criterions)
        else:
            stop = any(c(results) for c in criterions)
        return stop

    def evaluate(self, data, predictions=False):
        """

        :param data: list of LabeledPoint
        :param predictions: bool, for returning predictions too
        :return:
        """
        actual = map(lambda lp: lp.label, data)
        predicted = map(lambda lp: self.predict(lp.features).matrix(), data)
        if self.params.classification is True:
            # En problemas de clasificacion, se determina la prediccion por la unidad de softmax que predomina
            predicted = map(lambda p: float(np.argmax(p)), predicted)
        if self.params.classification is True:
            n_classes = self.params.units_layers[-1]  # La cant de unidades de la ult capa define la cant de clases
            metric = ClassificationMetrics(zip(predicted, actual), n_classes=n_classes)
            hits = metric.f_measure()
        else:
            metric = RegressionMetrics(zip(predicted, actual))
            hits = metric.r2()
        if predictions is True:  # Devuelvo ademas el vector de predicciones
            ret = hits, predicted
        else:
            ret = hits
        return ret

    def train(self, train_bc, mini_batch=50, parallelism=4, optimizer_params=None):
        """

        :param train_bc: sc.broadcast
        :param mini_batch:
        :param epochs:
        :param parallelism:
        :param optimizer_params:
        :return:
        """
        # TODO: ver improve_patience en deeplearning.net
        # Funciones a usar en los RDD
        minimizer = opt.optimize
        mixer = opt.mix_models
        # TODO ver si usar sc.accumulator para acumular actualizaciones y despues aplicarlas (paper de mosharaf)
        seeds = list(self.params.rng.randint(500, size=parallelism))
        # Paralelizo modelo actual en los nodos dados por parallelism
        models_rdd = sc.parallelize(zip([self] * parallelism, seeds))
        # Minimizo el costo de las redes en paralelo
        results = models_rdd.map(lambda (model, seed):
                                 minimizer(model, train_bc.value, mini_batch, optimizer_params, seed)).cache()
        if self.params.classification is True:
            layers = (results.map(lambda res: [layer * res['hits'] for layer in res['model']])
                             .reduce(lambda left, right: mixer(left, right)))
            # Se realiza un promedio ponderado por la tasa de aciertos en el train
            # TODO: y si todas las tasas son 0.0 ?? se divide por 0?!
            total_hits = results.map(lambda res: res['hits']).sum()
            final_list_layers = map(lambda layer: layer / total_hits, layers)
        else:
            layers = (results.map(lambda res: [layer for layer in res['model']])
                             .reduce(lambda left, right: mixer(left, right)))
            # Se realiza un promedio sin ponderacion
            final_list_layers = map(lambda layer: layer / parallelism, layers)
        self.list_layers = copy.copy(final_list_layers)
        results.unpersist()  # Saco de cache
        # Evaluo tasa de aciertos de entrenamiento
        hits = self.evaluate(train_bc.value)
        return hits

    def fit(self, train, valid=None, criterions=None, mini_batch=50, parallelism=4, optimizer_params=None,
            keep_best=False):
        # Creo Broadcasts, de manera de mandarlo una sola vez a todos los nodos
        train_bc = sc.broadcast(train)
        valid_bc = sc.broadcast(valid)  # Por ahora no es necesario el bc, pero puede q luego lo use en batchs p/ train
        # TODO: ver si hay mas opciones que quedarse con el mejor
        if keep_best is True:
            best_model = self.list_layers
            best_valid = 0.0
            best_train = 0.0
        if criterions is None:
            criterions = [criterion['MaxIterations'](5),
                          criterion['AchieveTolerance'](0.95, key='hits')]
        epoch = 0
        while self.check_stop(epoch, criterions) is False:
            self.hits_train = self.train(train_bc, mini_batch, parallelism, optimizer_params)
            self.hits_valid = self.evaluate(valid_bc.value)
            print "Epoca ", epoch+1, ". Hits en train: ", self.hits_train, ". Hits en valid: ", self.hits_valid
            if keep_best is True:
                if self.hits_valid >= best_valid:
                    best_valid = self.hits_valid
                    best_train = self.hits_train
                    best_model = self.list_layers
            epoch += 1
        if keep_best is True:
            self.list_layers = copy.copy(best_model)
            self.hits_train = best_train
            self.hits_valid = best_valid
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

    def persist_layers(self):
        for i in xrange(len(self.list_layers)):
            self.list_layers[i].persist_layer()

    def unpersist_layers(self):
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


class AutoEncoder(NeuralNetwork):
    def __init__(self, params=None, list_layers=None, sparsity_beta=0, sparsity_param=0.05):
        # Aseguro algunos parametros
        params.classification = False
        n_in = params.units_layers[0]
        params.units_layers.append(n_in) # Unidades en la salida en igual cantidad que la entrada
        params.dropout_ratios = [0.0] * len(params.units_layers)  # Sin dropout por ser regresion
        self.sparsity_beta = sparsity_beta
        self.sparsity_param = sparsity_param
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
            w_t = self.list_layers[-l + 1].weights_T
            delta = w_t.mul_array(delta).mul_elemwise(d_a[-l])
            nabla_w[-l] = delta.outer(a[-l - 1])
            nabla_b[-l] = delta
        return cost, (nabla_w, nabla_b)

    def evaluate(self, data, predictions=False):
        actual = map(lambda lp: lp.features, data)  # Tiene que aprender a reconstruir la entrada
        predicted = map(lambda lp: self.predict(lp.features).matrix().T, data)  # TODO notar que tuve q transponer
        metric = RegressionMetrics(zip(predicted, actual))
        hits = metric.r2()
        if predictions is True:  # Devuelvo ademas el vector de predicciones
            ret = hits, predicted
        else:
            ret = hits
        return ret

    def kl_divergence(self, x):
        pass

    def encode(self, x):
        if isinstance(x, list):
            x = map(lambda lp: self.encode(lp.features).matrix(), x)
        else:
            x = self.encoder_layer().output(x, grad=False)   # Solo la salida de la capa oculta
        return x

    def encoder_layer(self):
        return self.list_layers[0]

    def assert_regression(self):
        # Aseguro que el autoencoder tenga capas de regresion
        for l in xrange(self.num_layers):
            if type(self.list_layers[l]) is ClassificationLayer:
                layer = RegressionLayer()
                layer.__dict__ = self.list_layers[l].__dict__.copy()
                self.list_layers[l] = layer


class StackedAutoencoder(NeuralNetwork):

    def __init__(self, params=None, list_layers=None, sparsity_beta=0, sparsity_param=0.05):
        self.sparsity_beta = sparsity_beta
        self.sparsity_param = sparsity_param
        self.params = params
        self.num_layers = len(params.units_layers)
        NeuralNetwork.__init__(self, params, list_layers=None)
        self._init_ae()  # Creo autoencoders que se guardan en list_layers

    def _init_ae(self):
        for l in xrange(len(self.list_layers)):
            # Genero nueva estructura de parametros acorde al Autoencoder a crear
            params = DeepLearningParams(self.params.units_layers[l:l+2], activation=self.params.activation,
                                        layer_distributed=self.params.layer_distributed, dropout_ratios=None,
                                        classification=False, strength_l1=self.params.strength_l1,
                                        strength_l2=self.params.strength_l2)
            self.list_layers[l] = AutoEncoder(params=params, sparsity_beta=self.sparsity_beta,
                                              sparsity_param=self.sparsity_param)

    def fit(self, train, valid=None, criterions=None, mini_batch=50, parallelism=4, optimizer_params=None,
            keep_best=False):
        # Inicializo Autoencoders
        train_ae = train
        valid_ae = valid
        labels_train = map(lambda lp: lp.label, train_ae)
        labels_valid = map(lambda lp: lp.label, valid_ae)
        for ae in self.list_layers:
            print "Entrenando Autoencoder ", ae.params.units_layers
            ae.assert_regression()
            ae.fit(train_ae, valid_ae, criterions=criterions, mini_batch=mini_batch, parallelism=parallelism,
                   optimizer_params=optimizer_params, keep_best=keep_best)
            # Siguen siendo importantes los labels para el sample balanceado por clases
            train_ae = label_data(ae.encode(train_ae), labels_train)
            valid_ae = label_data(ae.encode(valid_ae), labels_valid)
        self.hits_valid = self.evaluate(valid)
        return self.hits_valid

    def finetune(self, train, valid, criterions=None, mini_batch=50, parallelism=4, optimizer_params=None,
                 keep_best=False):
        list_layers = map(lambda ae: ae.encoder_layer(), self.list_layers)  # Tomo solo la capa de encoder de cada ae
        nn = NeuralNetwork(self.params, list_layers=list_layers)
        hits_valid = nn.fit(train, valid, mini_batch=mini_batch, parallelism=parallelism, criterions=criterions,
                            optimizer_params=optimizer_params)
        for l in xrange(len(self.list_layers)):
            # Copio capa con ajuste fino al autoencoder
            self.list_layers[l].list_layers[0] = nn.list_layers[l]  # TODO mejorar esto, para que sea mas legible
        return hits_valid

    def predict(self, x):
        if isinstance(x, list):
            x = map(lambda lp: self.predict(lp.features).matrix(), x)
        else:
            for i in xrange(self.num_layers):
                x = self.list_layers[i].encode(x)
        return x