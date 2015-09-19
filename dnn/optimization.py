__author__ = 'leferrad'


import numpy as np
import loss
from neurons import LocalNeurons

class OptimizerParameters:
    def __init__(self, algorithm='Adadelta', num_epochs=100, mini_batch_size=100, options=None):
        if options is None:  # Agrego valores por defecto
            if algorithm == 'Adadelta':
                options = {'step-rate': 1, 'decay': 0.9, 'momentum': 0}
            elif algorithm == 'SGD':
                options = {'step-rate': 1, 'momentum': 0}
        self.options = options
        self.algorithm = algorithm
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size




# https://github.com/vitruvianscience/OpenDeep/blob/master/opendeep/optimization/optimizer.py
class Optimizer(object):
    """
    Default interface for an optimizer implementation - this provides the necessary parameter updates when
    training a model on a dataset using an online stochastic process. The base framework for performing
    stochastic gradient descent.
    """

    def __init__(self, model, data, parameters=None):
        self.model = model
        self.num_layers = model.num_layers
        self.data = data
        if parameters is None:
            parameters = OptimizerParameters()
        self.parameters = parameters
        self.n_epoch = 0

    def __iter__(self):
        for info in self._iterate():
            yield 'Epoca ' + str(info['n_epoch']) +'. Costo: ' + str(info['cost'])





class Adadelta(Optimizer):
    """Adadelta optimizer.
    https://github.com/BRML/climin/blob/master/climin/adadelta.py
    .. [zeiler2013adadelta] Zeiler, Matthew D.
       "ADADELTA: An adaptive learning rate method."
       arXiv preprint arXiv:1212.5701 (2012).
    """

    def __init__(self, model, data, parameters=None):
        """

        :param model: NeuralNetwork
        :param data: list of LabeledPoint
        :param parameters: OptimizerParameters
        :return:
        """
        super(Adadelta, self).__init__(model, data, parameters)
        self._init_acummulators()


    def _update(self):
        self.model.update(self.step_w, self.step_b)

    def _init_acummulators(self):
        """
        Inicializo acumuladores usados para la optimizacion
        :return:
        """
        self.gms_w = []
        self.gms_b = []
        self.sms_w = []
        self.sms_b = []
        self.step_w = []
        self.step_b = []
        for layer in self.model.list_layers:
            shape_w = layer.get_weights().shape()
            shape_b = layer.get_bias().shape()
            self.gms_w.append(LocalNeurons(np.zeros(shape_w), shape_w))
            self.gms_b.append(LocalNeurons(np.zeros(shape_b), shape_b))
            self.sms_w.append(LocalNeurons(np.zeros(shape_w), shape_w))
            self.sms_b.append(LocalNeurons(np.zeros(shape_b), shape_b))
            self.step_w.append(LocalNeurons(np.zeros(shape_w), shape_w))
            self.step_b.append(LocalNeurons(np.zeros(shape_b), shape_b))



    def _iterate(self):
        for ep in xrange(self.parameters.num_epochs):
            d = self.parameters.options['decay']
            o = 1e-4  # offset
            m = self.parameters.options['momentum']
            sr = self.parameters.options['step-rate']
            for lp in self.data:  # Por cada LabeledPoint del conj de datos
                # 1) Computar el gradiente
                cost, (nabla_w, nabla_b) = self.model.cost(lp.features, lp.label)
                for l in xrange(self.num_layers):
                    # ADICIONAL: Aplico momentum y step-rate (ANTE LA DUDA, COMENTAR ESTAS LINEAS)
                    step1w = self.step_w[l] * m * sr
                    step1b = self.step_b[l] * m * sr
                    # 2) Acumular el gradiente
                    self.gms_w[l] = (self.gms_w[l] * d) + (nabla_w[l] ** 2) * (1 - d)
                    self.gms_b[l] = (self.gms_b[l] * d) + (nabla_b[l] ** 2) * (1 - d)
                    # 3) Computar actualizaciones
                    step2w = ((self.sms_w[l] + o) ** 0.5) / ((self.gms_w[l] + o) ** 0.5) * nabla_w[l] * sr
                    step2b = ((self.sms_b[l] + o) ** 0.5) / ((self.gms_b[l] + o) ** 0.5) * nabla_b[l] * sr
                    # 4) Acumular actualizaciones
                    self.step_w[l] = step1w + step2w
                    self.step_b[l] = step1b + step2b
                    self.sms_w[l] = (self.sms_w[l] * d) + (self.step_w[l] ** 2) * (1 - d)
                    self.sms_b[l] = (self.sms_b[l] * d) + (self.step_b[l] ** 2) * (1 - d)
                # 5) Aplicar actualizaciones a todas las capas
                self._update()

            self.n_epoch += 1
            yield {
                'n_epoch': self.n_epoch,
                'cost': cost
            }

class SGD:
    def __init__(self, train_data, epochs, mini_batch_size,
                 momentum, valid_data=None):
        self.train_data = train_data
        self.valid_data = valid_data
        self.mini_batch_size = mini_batch_size




        if valid_data: n_test = len(valid_data)
        n = len(train_data)
        for j in xrange(epochs):
            mini_batches = [
                train_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, momentum)
            if valid_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(valid_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


Minimizer = {'Adadelta': Adadelta}