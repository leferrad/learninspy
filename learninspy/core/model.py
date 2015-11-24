__author__ = 'leferrad'

# Dependencias externas
import numpy as np
from scipy import sparse

# Dependencias internas
from learninspy.core import activations as act, loss, optimization as opt
from learninspy.core.stops import criterion
from learninspy.core.neurons import DistributedNeurons, LocalNeurons
from learninspy.utils import checks
from learninspy.utils.evaluation import ClassificationMetrics, RegressionMetrics
from learninspy.utils.data import LabeledDataSet
from learninspy.context import sc

# Librerias de Python
import copy
import cPickle as pickle
import os


class NeuralLayer(object):

    def __init__(self, n_in=2, n_out=2, activation='ReLU', distribute=False, w=None, b=None, sparsity=False, rng=None):
        self.n_out = n_out
        self.n_in = n_in
        self.activation = act.fun_activation[activation]
        self.activation_d = act.fun_activation_d[activation]
        self.sparsity = False  # TODO completar esta funcionalidad

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

        # TODO weights_T era p/ poder hacer operaciones distribuidas, pero se deja como experimental DistributedNeurons
        distribute = False
        if distribute is True:
            self.weights = DistributedNeurons(w, self.shape_w)
            #self.weights_T = DistributedNeurons(w.T, self.shape_w[::-1])
            self.bias = DistributedNeurons(b, self.shape_b)
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
class NetworkParameters:
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
        config += "The loss is "+self.loss+" for a problem of "
        if self.classification is True:
            config += "classification."+os.linesep
        else:
            config += "regression."+os.linesep
        config += "Strength in L1 norm is "+str(self.strength_l1) + \
                  " and in L2 norm is "+str(self.strength_l2)+"."+os.linesep
        return config



class NeuralNetwork(object):
    def __init__(self, params, list_layers=None):
        self.params = params
        self.list_layers = list_layers  # En caso de que la red reciba capas ya inicializadas
        self.loss = loss.fun_loss[self.params.loss]
        self.loss_d = loss.fun_loss_d[self.params.loss]

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
            w_t = self.list_layers[-l + 1].weights.transpose()
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

        :param train_bc:
        :param mini_batch:
        :param parallelism:
        :param optimizer_params:
        :return:
        """
        # Funcion para minimizar funcion de costo sobre cada modelo del RDD
        minimizer = opt.optimize
        # TODO ver si usar sc.accumulator para acumular actualizaciones y despues aplicarlas (paper de mosharaf)
        seeds = list(self.params.rng.randint(500, size=parallelism))
        # Paralelizo modelo actual en los nodos dados por parallelism
        models_rdd = sc.parallelize(zip([self] * parallelism, seeds))
        # Minimizo el costo de las redes en paralelo
        # NOTA: cache() es importante porque se traza varias veces el grafo de acciones sobre el RDD results
        results = models_rdd.map(lambda (model, seed):
                                 minimizer(model, train_bc.value, mini_batch, optimizer_params, seed)).cache()
        # Junto modelos entrenados en paralelo, en base a un criterio de ponderacion sobre un valor objetivo
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
            keep_best=False):
        # Si son LabeledDataSet, los colecto en forma de lista
        if type(train) is LabeledDataSet:
            train = train.collect()
        if type(valid) is LabeledDataSet:
            valid = valid.collect()
        # Creo Broadcasts, de manera de mandarlo una sola vez a todos los nodos
        train_bc = sc.broadcast(train)
        valid_bc = sc.broadcast(valid)  # Por ahora no es necesario el bc, pero puede q luego lo use en batchs p/ train
        # TODO: ver si hay mas opciones que quedarse con el mejor
        if keep_best is True:
            best_model = self.list_layers
            best_valid = 0.0
            best_train = 0.0
        if stops is None:
            stops = [criterion['MaxIterations'](5),
                     criterion['AchieveTolerance'](0.95, key='hits')]
        epoch = 0
        while self.check_stop(epoch+1, stops) is False:
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


