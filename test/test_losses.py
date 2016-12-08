#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.loss import fun_loss
from learninspy.utils.fileio import get_logger
from learninspy.utils.checks import CheckGradientLoss

logger = get_logger(name=__name__)

def test_losses_gradients():
    logger.info("Testeando gradientes implementados para las funciones de error...")
    for loss in fun_loss.keys():  # Dado que hay una regla de cadena de por medio, voy a tener que saltear este check
        if loss == 'CrossEntropy':  # TODO: ver si conviene el check en esta función mediante deriv por regla de cadena
            continue
        logger.info("Gradiente de función %s:", loss)
        check = CheckGradientLoss(loss)
        good_grad = check()
        assert good_grad, AssertionError("Gradiente de función "+loss+" mal implementado!")
        logger.info("OK.")
