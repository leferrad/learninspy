__author__ = 'leferrad'


import numpy as np
import loss


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

    def __init__(self, data, layers, dropout_ratios=None, loss_func='MSE', parameters=None):
        self.data = data
        self.list_layers = layers
        self.num_layers = len(layers)
        if dropout_ratios is None:
            dropout_ratios = [0.0] * self.num_layers
        self.dropout_ratios = dropout_ratios
        self.loss = loss.fun_loss[loss_func]
        self.loss_d = loss.fun_loss_d[loss_func]
        if parameters is None:
            parameters = OptimizerParameters()
        self.parameters = parameters
        self.n_epoch = 0

    def func(self, x, y):
        # func es el backpropagation
        # Ver http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
        num_layers = self.num_layers
        drop_fraction = self.dropout_ratios  # Vector con las fracciones de DropOut para cada NeuralLayer
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

    def update(self, step_w, step_b, index):
        step_w = -step_w
        step_b = -step_b
        self.list_layers[index].update(step_w, step_b)
        return



class Adadelta(Optimizer):
    """Adadelta optimizer.
    https://github.com/BRML/climin/blob/master/climin/adadelta.py
    .. [zeiler2013adadelta] Zeiler, Matthew D.
       "ADADELTA: An adaptive learning rate method."
       arXiv preprint arXiv:1212.5701 (2012).
    """

    def __init__(self, data, layers, dropout_ratios=None, loss_func='MSE', parameters=None):
        """Create an Adadelta object.

        """
        Optimizer.__init__(self, data, layers, dropout_ratios, loss_func, parameters)

        self.gms_w = 0
        self.gms_b = 0
        self.sms = 0
        self.step = 0

    def _iterate(self):
        d = self.parameters.options['decay']
        o = 1e-4  # offset
        m = self.parameters.options['momentum']
        sr = self.parameters.options['step-rate']
        for lp in self.data:  # Por cada LabeledPoint del conj de datos
            # 1) Computar el gradiente
            cost, (nabla_w, nabla_b) = self.func(lp.features, lp.label)
            for l in xrange(self.num_layers):
                # ADICIONAL: Aplico momentum y step-rate (ANTE LA DUDA, COMENTAR ESTAS LINEAS)
                step1w = self.stepw * m * sr
                step1b = self.stepb * m * sr
                self.update(step1w, step1b, l)
                # 2) Acumular el gradiente
                self.gms_w = (self.gms_w * d) + (nabla_w[l] ** 2) * (1 - d)
                self.gms_b = (self.gms_b * d) + (nabla_b[l] ** 2) * (1 - d)
                # 3) Computar actualizaciones
                step2w = ((self.sms + o) ** 0.5) / ((self.gms_w + o) ** 0.5) * nabla_w[l] * sr
                step2b = ((self.sms + o) ** 0.5) / ((self.gms_b + o) ** 0.5) * nabla_b[l] * sr
                # 4) Aplicar actualizaciones
                self.update(step2w, step2b, l)
                self.stepw = step1w + step2w
                self.stepb = step1b + step2b
                # 5) Acumular actualizaciones
                self.smsw = (self.smsw * d) + (self.stepw ** 2) * (1 - d)
                self.smsb = (self.smsb * d) + (self.stepb ** 2) * (1 - d)

        self.n_epoch += 1
        pass

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