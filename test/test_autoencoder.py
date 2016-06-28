#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

import os
import sys


# Append pyspark  to Python Path
sys.path.append(os.environ['SPARK_HOME']+"/python/")
sys.path.append(os.environ['SPARK_HOME']+"/python/lib/py4j-0.9-src.zip")  # Esta soluciona el problema con py4j


def test_autoencoder():
    assert 1.0 == 1.0  # TODO: hacer esto
