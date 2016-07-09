#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.activations import fun_activation
from learninspy.utils.checks import CheckGradientActivation


def test_activations_gradients():
    check = CheckGradientActivation(fun_activation.keys())  # Chequeo de todas las funciones de activ implementadas
    bad_gradients = check()
    assert bad_gradients is None
