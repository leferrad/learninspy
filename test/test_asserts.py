#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.utils.asserts import *
from learninspy.core.neurons import LocalNeurons
from learninspy.utils.fileio import get_logger

import numpy as np

logger = get_logger(name=__name__)


class TestAsserts(object):
    def __init__(self):
        pass

    # -- Calls --
    @staticmethod
    @assert_matchdimension
    def test_matchdimension(arr1, arr2):
        return

    @staticmethod
    @assert_samedimension
    def test_samedimension(arr1, arr2):
        return

    @staticmethod
    @assert_sametype
    def test_sametype(arr1, arr2):
        return

    @assert_features_label
    def test_features_label(self, data):
        return

    def test_all(self):
        # 1) Test de assert_features_label

        # 1.a) Sin exception
        # Dummy data
        features = np.array(range(10))
        label = 1
        data = (features, label)
        test_ok = True
        try:
            self.test_features_label(data)
            test_ok = True
        except:
            test_ok = False

        assert test_ok

        # 1.b) Con exception
        try:
            self.test_features_label(features)
            test_ok = False  # Tendría que arrojar un Exception por recibir un arreg de dim 10
        except:
            test_ok = True

        assert test_ok

        # 1.c) No debe actuar con None
        try:
            self.test_features_label(None)
            test_ok = True
        except:
            test_ok = False

        assert test_ok

        # 2) Test de assert_samedimension

        # 2.a) Sin exception
        try:
            self.test_samedimension(features, features)
            test_ok = True
        except:
            test_ok = False

        assert test_ok

        # 2.b) Con exception
        try:
            self.test_samedimension(features, features[:5])  # Dimensiones diferentes
            test_ok = False
        except:
            test_ok = True

        assert test_ok

        # 3) Test de assert_matchdimension

        # Matrices de neuronas
        shape = (5, 10)
        matrix1 = LocalNeurons(np.zeros(shape), shape=shape)

        shape = (10, 3)
        matrix2 = LocalNeurons(np.zeros(shape), shape=shape)

        # 3.a) Sin exception
        try:
            self.test_matchdimension(matrix1, matrix2)
            test_ok = True
        except:
            test_ok = False

        assert test_ok

        # 3.b) Con exception
        try:
            self.test_matchdimension(matrix2, matrix1)  # Dimensiones no compatibles
            test_ok = False
        except:
            test_ok = True

        assert test_ok