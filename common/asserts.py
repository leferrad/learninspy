__author__ = 'leferrad'

# Dependencias externas
import pyspark.rdd
from pyspark.mllib.regression import LabeledPoint
import numpy as np

# Decoradores para validacion:
def assert_typearray(func):
    def func_assert(*args):
        # arg[0] es self y arg[1] es array
        if not isinstance(args[1], type(args[0])):
            raise Exception('Solo se puede operar con arreglos del mismo tipo!')
        else:
            return func(*args)
    return func_assert


def assert_samedimension(func):
    def func_assert(*args):
        # arg[0] es self y arg[1] es array
        if args[0].shape() != args[1].shape():
            raise Exception('Arreglos con dimensiones distintas!')
        else:
            return func(*args)
    return func_assert


def assert_matchdimension(func):
    def func_assert(*args):
        # arg[0] es self y arg[1] es array
        if args[0].cols != args[1].rows:
            raise Exception('Arreglos con dimensiones que no coinciden!')
        else:
            return func(*args)
    return func_assert


def assert_features_label(func):
    def func_assert(*args):
        # args es list o np.array o RDD o None
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
            # Evaluo si la entrada es de dimension 2 (label-features)
            if entry is not None:
                if type(entry) is LabeledPoint:
                    ok = True
                elif len(entry) != 2:
                    ok = False
            # Si ok es True, se respeto la estructura
            if ok is False:
                raise Exception('Dataset no respeta estructura features-label! (dim!=2)')
            else:
                return func(*args)
    return func_assert