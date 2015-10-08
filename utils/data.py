__author__ = 'leeandro04'

import random
import numpy as np
import pyspark.rdd
from pyspark.mllib.regression import LabeledPoint

def label_to_vector(label, n_classes):
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)


def subsample(data, size, balanced=True, seed=123):
    """
    Muestreo de data, con resultado balanceado por clases si se lo pide
    :param data: list of LabeledPoint
    :param size: int
    :param seed: int
    :return:

    """
    random.seed(seed)
    if balanced is True:  #Problema de clasificacion
        n_classes = int(max(map(lambda lp: lp.label, data))) + 1
        size = size / n_classes  # es un int, y puede resultar menor al ingresado (se trunca)
        sample = []
        for c in xrange(n_classes):
            batch_class = filter(lambda lp: lp.label == c, data)  # Filtro entradas que pertenezcan a la clase c
            batch = random.sample(batch_class, size)
            sample.extend(batch)  # Agrego el batch al vector de muestreo
    else:  #Problema de regresion
        sample = random.sample(data, size)
    random.shuffle(sample)  # Mezclo para agregar aleatoriedad
    return sample


def split_data(data, fractions, seed=123):
    """
    Split data en sets en base a fractions
    :param data: list or np.array
    :param fractions: list [f_train, f_valid, f_test]
    :param seed: int
    :return: sets (e.g. train, valid, test)
    """
    # Verifico que fractions sea correcto
    # TODO: assert (sum(fractions) <= 1.0, Exception("Fracciones para conjuntos incorrectas!"))
    if isinstance(data, pyspark.rdd.PipelinedRDD):
        sets = data.randomSplit(fractions, seed)  # Uso la funcion del RDD
    else:
        # Mezclo un poco los datos
        random.seed(seed)
        random.shuffle(data)
        # Segmento conjuntos
        size_data = len(data)
        size_split = map(lambda f: int(size_data * f), fractions)
        index_split = [0]+ size_split[:-1]
        sets = [data[i:i+size] for i, size in zip(index_split, size_split)]
    return sets


def label_data(data, label):
    labeled_data = map(lambda (x, y): LabeledPoint(y, x), zip(data, label))
    return labeled_data