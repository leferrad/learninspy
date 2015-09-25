__author__ = 'leferrad'

import logging
import numpy as np
import random

class LearninspyLogger(object):
    def __init__(self, level='INFO'):
        logger = logging.getLogger('Learninspy')
        if level == 'INFO':
            logger.setLevel(logging.INFO)
        elif level == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARN)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger = logger
        self.logger.addHandler(ch)

        # 'application' code
        self.logger.debug('debug message')
        self.logger.info('info message')
        self.logger.warn('warn message')
        self.logger.error('error message')
        self.logger.critical('critical message')

    def info(self, msg):
        self.logger.info(msg)

def label_to_vector(label, n_classes):
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)

def balanced_subsample(data, size, seed=123):
    """
    Muestreo de data, con resultado balanceado por clases
    :param data: list of LabeledPoint
    :param size: int
    :param seed: int
    :return:

    """
    random.seed(seed)
    n_classes = int(max(map(lambda lp: lp.label, data))) + 1
    size = size / n_classes  # es un int, y puede resultar menor al ingresado (se trunca)
    sample = []
    for c in xrange(n_classes):
        batch_class = filter(lambda lp: lp.label == c, data)  # Filtro entradas que pertenezcan a la clase c
        batch = random.sample(batch_class, size)
        sample.extend(batch)  # Agrego el batch al vector de muestreo
    random.shuffle(sample)  # Mezclo para que no este segmentado por clases
    return sample