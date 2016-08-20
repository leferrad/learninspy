#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Decoradores para validar argumentos de funciones."""

__author__ = 'leferrad'

import pyspark.rdd
from pyspark.mllib.regression import LabeledPoint
import numpy as np


def assert_sametype(func):
    """
    Decorador de función para asegurar el mismo tipo de variable.
    """
    def func_assert(*args):
        # arg[0] es self y arg[1] es array
        if not isinstance(args[1], type(args[0])):
            raise Exception('Arreglos de distinto tipo!')
        else:
            return func(*args)
    return func_assert


def assert_samedimension(func):
    """
    Decorador de función para asegurar iguales dimensiones en arreglos.
    """
    def func_assert(*args):
        if args[0].shape != args[1].shape:
            raise Exception('Arreglos con dimensiones distintas!')
        else:
            return func(*args)
    return func_assert


def assert_matchdimension(func):
    """
    Decorador de función para asegurar dimensiones compatibles en producto matricial.
    La validación se realiza sobre un objeto del tipo :class:`learninspy.utils.data.LabeledDataSet`
    o del tipo :class:`learninspy.core.neurons.LocalNeurons`.
    """
    def func_assert(*args):
        array1 = args[0]
        array2 = args[1]

        try:
            n_cols_0 = len(array1[0])
        except:
            # it should be a LocalNeurons (can't import it due to a cyclic calling)
            n_cols_0 = array1.cols

        try:
            n_rows_1 = len(array2)
        except:
            # it should be a LocalNeurons (can't import it due to a cyclic calling)
            n_rows_1 = array2.rows

        if n_cols_0 != n_rows_1:
            raise Exception('Arreglos con dimensiones que no coinciden!'+str(n_cols_0)+","+str(n_rows_1))
        else:
            return func(*args)
    return func_assert


def assert_features_label(func):
    """
    Decorador de función para asegurar estructura Features-Label en una instancia
    de :class:`learninspy.utils.data.LabeledDataSet`.
    """
    def func_assert(*args):
        # La idea es mirar si el arg referido a data tiene dimensión 2,
        # lo cual supone una estructura features-label.
        if len(args) == 1:  # data es None, no hay nada que chequear
            return func(args[0])
        else:
            data = args[1]
            ok = True  # Flag para indicar que el assert da OK
            entry = None
            # Extraigo una entrada de data p/ evaluar estructura
            if data is not None:
                if isinstance(data, pyspark.rdd.RDD):
                    entry = data.take(1)[0]
                elif type(data) is np.ndarray or type(data) is list:
                    entry = data[0]
            # Evalúo si la entrada es de dimensión 2 (features-label)
            if entry is not None:
                if type(entry) is LabeledPoint:
                    ok = True
                elif len(entry) != 2:
                    ok = False
            # Si ok es True, se respetó la estructura
            if ok is False:
                raise Exception('Argumento no respeta estructura features-label! (dim!=2)')
            else:
                return func(*args)
    return func_assert