__author__ = 'leferrad'


import numpy as np
import loss
from neurons import LocalNeurons
import copy

class OptimizerParameters:
    def __init__(self, algorithm='Adadelta', num_epochs=10, tolerance=0.99, options=None):
        if options is None:  # Agrego valores por defecto
            if algorithm == 'Adadelta':
                options = {'step-rate': 1, 'decay': 0.99, 'momentum': 0.0, 'offset': 1e-8}
            elif algorithm == 'GD':
                options = {'step-rate': 1, 'momentum': 0.3, 'momentum_type': 'standart'}
        self.options = options
        self.algorithm = algorithm
        self.num_epochs = num_epochs
        self.tolerance = tolerance




# https://github.com/vitruvianscience/OpenDeep/blob/master/opendeep/optimization/optimizer.py
class Optimizer(object):
    """
    Default interface for an optimizer implementation - this provides the necessary parameter updates when
    training a model on a dataset using an online stochastic process. The base framework for performing
    stochastic gradient descent.
    """

    def __init__(self, model, data, parameters=None):
        self.model = copy.copy(model)
        self.num_layers = model.num_layers
        self.data = data
        if parameters is None:
            parameters = OptimizerParameters()
        self.parameters = parameters
        self.cost = 0.0
        self.n_epoch = 0
        self.hits = 0.0
        self.step_w = None
        self.step_b = None

    def _iterate(self):
        # Implementacion hecha en las clases que heredan
        yield

    def _update(self):
        self.model.update(self.step_w, self.step_b)

    def __iter__(self):
        for info in self._iterate():
            continue
        yield {
            'model': self.model.list_layers,
            'hits': self.hits,
            'epochs': self.n_epoch,
            'cost': self.cost
        }

    def check_criterion(self):
        epochs = self.n_epoch >= self.parameters.num_epochs
        tolerance = self.hits >= self.parameters.tolerance
        return epochs or tolerance





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
        while self.check_criterion() is False:
            d = self.parameters.options['decay']
            o = self.parameters.options['offset']  # offset
            m = self.parameters.options['momentum']
            sr = self.parameters.options['step-rate']
            # --- Entrenamiento ---
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
                    self.step_w[l] = (step1w + step2w) * -1.0
                    self.step_b[l] = (step1b + step2b) * -1.0
                    self.sms_w[l] = (self.sms_w[l] * d) + (self.step_w[l] ** 2) * (1 - d)
                    self.sms_b[l] = (self.sms_b[l] * d) + (self.step_b[l] ** 2) * (1 - d)
                # 5) Aplicar actualizaciones a todas las capas
                self.cost = cost
                self._update()
            # --- Error de clasificacion---
            data = copy.deepcopy(self.data)
            self.hits = self.model.evaluate(data)
            self.n_epoch += 1
            yield {
                'n_epoch': self.n_epoch,
                'hits': self.hits,
                'cost': self.cost
            }


class GD(Optimizer):
    def __init__(self, model, data, parameters=None):
        super(GD, self).__init__(model, data, parameters)
        self._init_acummulators()

    def _init_acummulators(self):
        """
        Inicializo acumuladores usados para la optimizacion
        :return:
        """
        self.step_w = []
        self.step_b = []
        for layer in self.model.list_layers:
            shape_w = layer.get_weights().shape()
            shape_b = layer.get_bias().shape()
            self.step_w.append(LocalNeurons(np.zeros(shape_w), shape_w))
            self.step_b.append(LocalNeurons(np.zeros(shape_b), shape_b))

    def _iterate(self):
        while self.check_criterion() is False:
            m = self.parameters.options['momentum']
            sr = self.parameters.options['step-rate']
            # --- Entrenamiento ---
            for lp in self.data:  # Por cada LabeledPoint del conj de datos
                # 1) Computar el gradiente
                cost, (nabla_w, nabla_b) = self.model.cost(lp.features, lp.label)
                for l in xrange(self.num_layers):
                    if self.parameters.options['momentum_type'] == 'standard':
                        self.step_w[l] = nabla_w[l] * sr + self.step_w[l] * m
                        self.step_b[l] = nabla_b[l] * sr + self.step_b[l] * m
                    elif self.parameters.options['momentum_type'] == 'nesterov':
                        big_jump_w = self.step_w[l] * m
                        big_jump_b = self.step_b[l] * m

                        correction_w = nabla_w[l] * sr
                        correction_b = nabla_b[l] * sr

                        self.step_w[l] = big_jump_w + correction_w
                        self.step_b[l] = big_jump_b + correction_b

                # Aplicar actualizaciones a todas las capas
                self.cost = cost
                self._update()
            # --- Error de clasificacion---
            data = copy.deepcopy(self.data)
            self.hits = self.model.evaluate(data)
            self.n_epoch += 1
            yield {
                'n_epoch': self.n_epoch,
                'hits': self.hits,
                'cost': self.cost
            }

Minimizer = {'Adadelta': Adadelta, 'GD': GD}