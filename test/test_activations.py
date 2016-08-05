#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.activations import fun_activation
from learninspy.utils.checks import CheckGradientActivation


def test_activations_gradients():
    # TODO: logging
    for act in fun_activation.keys():
        check = CheckGradientActivation(act)  # Chequeo de todas las funciones de activ implementadas
        good_grad = check()
        assert good_grad, AssertionError("Gradiente de función "+act+" mal implementado!")

test_activations_gradients()