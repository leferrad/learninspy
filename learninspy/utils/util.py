__author__ = 'leferrad'

import logging

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

    def info(self, msg):
        self.logger.info(msg)

