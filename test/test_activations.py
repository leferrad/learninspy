#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.activations import fun_activation
from learninspy.utils.fileio import get_logger
from learninspy.utils.checks import CheckGradientActivation

logger = get_logger(name=__name__)

def test_activations_gradients():
    logger.info("Testeando gradientes implementados para las funciones de activación...")
    for act in fun_activation.keys():
        logger.info("Gradiente de función %s:", act)
        check = CheckGradientActivation(act)
        good_grad = check()
        assert good_grad, AssertionError("Gradiente de función "+act+" mal implementado!")
        logger.info("OK.")