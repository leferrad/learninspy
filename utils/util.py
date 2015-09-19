__author__ = 'leferrad'

import logging
import numpy as np

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

def label_to_vector(label, nclasses):
    lab = np.zeros((nclasses, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)