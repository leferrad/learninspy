#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.loss import fun_loss
from learninspy.utils.checks import CheckGradientLoss


def test_losses_gradients():
    # TODO: logging
    for loss in fun_loss.keys():
        check = CheckGradientLoss(loss)  # Chequeo de todas las funciones de error implementadas
        good_grad = check()
        #assert good_grad, AssertionError("Gradiente de función "+loss+" mal implementado!")

test_losses_gradients()
