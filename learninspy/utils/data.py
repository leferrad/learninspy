#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Módulo destinado al tratamiento de datos, en la construcción y procesamiento de datasets."""

__author__ = 'leferrad'

from learninspy.utils import fileio
from learninspy.context import sc
from learninspy.utils.asserts import assert_features_label

import numpy as np
import pyspark.rdd
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler as StdSc

import cPickle
import gzip
import random
import os
import abc
from operator import add


# -- Normalización de datos --

class StandardScaler(object):
    """
    Estandariza un conjunto de datos, mediante la sustracción de la media y el escalado para tener varianza unitaria.
    Soporta RDDs usando la clase
    `StandardScaler <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.feature.StandardScaler>`_
    de **pyspark.mllib**.

    :param mean: bool, para indicar que se desea centrar conjunto de datos restándole la media.
    :param std: bool, para indicar que se desea normalizar conjunto de datos diviendo por el desvío estándar.

    >>> train = np.array([[-2.0, 2.3, 0.0], [3.8, 0.0, 1.9]])
    >>> test = np.array([[-1.0, 1.3, -0.5], [1.8, 2.2, -1.5]])
    >>> standarizer = StandardScaler(mean=True, std=True)
    >>> standarizer.fit(train)
    >>> standarizer.transform(train)
    array([[-0.70710678,  0.70710678, -0.70710678],[ 0.70710678, -0.70710678,  0.70710678]])
    >>> standarizer.transform(test)
    array([[-0.46327686,  0.09223132, -1.07926824],[ 0.21944693,  0.64561923, -1.82359117]])

    """
    def __init__(self, mean=True, std=True):
        self.flag_mean = mean
        self.flag_std = std
        self.mean = None
        self.std = None
        self.model = None

    def fit(self, dataset):
        """
        Computa la media y desvio estándar de un conjunto de datos, las cuales se usarán para estandarizar datos.

        :param dataset: pyspark.rdd.RDD o numpy.ndarray o :class:`.LabeledDataSet`

        """
        if isinstance(dataset, LabeledDataSet):
            dataset = dataset.features
        if isinstance(dataset, pyspark.rdd.RDD):
            standarizer = StdSc(self.flag_mean, self.flag_std)
            self.model = standarizer.fit(dataset)
        else:
            if type(dataset) is not np.ndarray:
                dataset = np.array(dataset)
            if self.flag_mean is True:
                self.mean = dataset.mean(axis=0)
            if self.flag_std is True:
                self.std = dataset.std(axis=0, ddof=1)
        return

    def transform(self, dataset):
        """
        Aplica estandarización sobre **dataset**.

        :param dataset: pyspark.rdd.RDD o numpy.ndarray o :class:`.LabeledDataSet`

        """
        labels = None  # Por si el dataset viene con labels
        type_dataset = type(dataset)
        if isinstance(dataset, LabeledDataSet):
            labels = dataset.labels
            dataset = dataset.features
        if self.model is not None:  # Se debe usar modelo de pyspark
            if not isinstance(dataset, pyspark.rdd.RDD):
                dataset = sc.parallelize(dataset)
            dataset = self.model.transform(dataset)
        else:
            if type(dataset) is not np.ndarray:
                dataset = np.array(dataset)
            if self.flag_mean is True:
                dataset -= self.mean  # Remove mean
            if self.flag_std is True:
                dataset /= self.std  # Scale unit variance
        if labels is not None:
            if type_dataset is DistributedLabeledDataSet:
                dataset = DistributedLabeledDataSet(labels.zip(dataset))
            else:
                dataset = LocalLabeledDataSet(zip(labels, dataset))
        return dataset

''''
class MinMaxScaler(object):
    def __init__(self, min=0.0, max=1.0):
        self.min = min
        self.max = max
        self.min_train = None
        self.max_train = None
        self.model = None

    def fit(self, dataset):
        """
        Computa la media y desvio estándar de un conjunto de datos, las cuales se usarán para estandarizar datos.

        :param dataset: pyspark.rdd.RDD o numpy.ndarray o :class:`.LabeledDataSet`

        """
        if isinstance(dataset, LabeledDataSet):
            dataset = dataset.features
        if isinstance(dataset, pyspark.rdd.RDD):
            standarizer = StdSc(self.flag_mean, self.flag_std)
            self.model = standarizer.fit(dataset)
        else:
            if type(dataset) is not np.ndarray:
                dataset = np.array(dataset)
            if self.flag_mean is True:
                self.mean = dataset.mean(axis=0)
            if self.flag_std is True:
                self.std = dataset.std(axis=0, ddof=1)
        return

    def transform(self, dataset):
        """
        Aplica estandarización sobre **dataset**.

        :param dataset: pyspark.rdd.RDD o numpy.ndarray o :class:`.LabeledDataSet`

        """
        labels = None  # Por si el dataset viene con labels
        type_dataset = type(dataset)
        if isinstance(dataset, LabeledDataSet):
            labels = dataset.labels
            dataset = dataset.features
        if self.model is not None:  # Se debe usar modelo de pyspark
            if not isinstance(dataset, pyspark.rdd.RDD):
                dataset = sc.parallelize(dataset)
            dataset = self.model.transform(dataset)
        else:
            if type(dataset) is not np.ndarray:
                dataset = np.array(dataset)
            if self.flag_mean is True:
                dataset -= self.mean  # Remove mean
            if self.flag_std is True:
                dataset /= self.std  # Scale unit variance
        if labels is not None:
            if type_dataset is DistributedLabeledDataSet:
                dataset = DistributedLabeledDataSet(labels.zip(dataset))
            else:
                dataset = LocalLabeledDataSet(zip(labels, dataset))
        return dataset
'''

# -- Clases para datasets --

class LabeledDataSet(object):
    """
    Clase base para construcción de datasets.
    """
    @abc.abstractmethod
    def features(self):
        """Devuelve sólo las características del conjunto de datos, en el correspondiente orden almacenado."""

    @abc.abstractmethod
    def labels(self):
        """Devuelve sólo las etiquetas del conjunto de datos, en el correspondiente orden almacenado."""

    @abc.abstractmethod
    def load_file(self, path, pos_label=-1):
        """Carga de conjunto de datos desde archivo."""

    @abc.abstractmethod
    def save_file(self, path):
        """Guardar conjunto de datos en archivo de texto."""

    @abc.abstractmethod
    def collect(self, **kwargs):
        """Devuelve el conjunto de datos,"""

    @abc.abstractmethod
    def shape(self):
        """Devuelve el tamaño del conjunto de datos alojado."""


class DistributedLabeledDataSet(LabeledDataSet):
    """
    Clase útil para manejar un conjunto etiquetado de datos. Dicho conjunto se almacena
    como un `pyspark.rdd.RDD <https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD>`_
    donde cada entrada posee un *pyspark.mllib.regression.LabeledPoint*.
    Se proveen funcionalidades para manejo de archivos, así como para partir el conjunto de datos
    (e.g. train, valid y test).

    :param data: list o numpy.ndarray o pyspark.rdd.RDD, o bien *None* si se desea iniciar un conjunto vacío.
    """
    @assert_features_label
    def __init__(self, data=None):
        self.with_lp = False  # Flag que indica si se almaceno entradas en forma de LabeledPoints
        if data is not None:
            if not isinstance(data, pyspark.rdd.RDD):
                data = sc.parallelize(data)
            if type(data.take(2)[0]) is LabeledPoint:
                self.with_lp = True
        self.data = data
        if self.with_lp is False and self.data is not None:  # Por ahora que quede etiquetado con LabeledPoints
            self._labeled_point()

        if self.data is not None:
            self.rows = self.data.count()
            self.cols = len(self.features.take(1)[0].toArray())

    @property
    def features(self):
        """
        Devuelve sólo las características del conjunto de datos, en el correspondiente orden almacenado.

        :return: pyspark.rdd.RDD
        """
        if self.with_lp is True:
            features = self.data.map(lambda lp: lp.features)
        else:
            features = self.data.map(lambda (l, f): f)
        return features

    @property
    def labels(self):
        """
        Devuelve sólo las etiquetas del conjunto de datos, en el correspondiente orden almacenado.

        :return: pyspark.rdd.RDD
        """
        if self.with_lp is True:
            labels = self.data.map(lambda lp: lp.label)
        else:
            labels = self.data.map(lambda (l, f): l)
        return labels

    def load_file(self, path, pos_label=-1):
        """
        Carga de conjunto de datos desde archivo. El formato aceptado es de archivos de texto, como CSV, donde
        los valores se separan por un caracter delimitador
        (configurable en :func:`~learninspy.utils.fileio.parse_point`).


        :param path: string, indicando la ruta de donde cargar los datos.
        :param pos_label: int, posición o n° de elemento de cada línea del archivo, que corresponde al **label** (por defecto es -1, que corresponde a la última posición).
        """
        self.data = fileio.load_file_spark(path, pos_label=pos_label)
        self._labeled_point()

        self.rows = self.data.count()
        self.cols = len(self.features.take(1)[0].toArray())

    def save_file(self, path):
        """
        Guardar conjunto de datos en archivo de texto.

        :param path: string, indicando la ruta en donde se guardan los datos.
        """
        if self.with_lp is True:
            data = self.data.map(lambda lp: (lp.label, lp.features))
        else:
            data = self.data
        data = data.map(lambda (label, features): [label] + list(features))
        fileio.save_file_spark(data, path)

    def _labeled_point(self):
        """
        Función para transformar cada entrada del conjunto de datos, de LabeledPoint a tupla o viceversa.

        """
        if self.with_lp is False:
            if self.data is not None:
                self.data = self.data.map(lambda (l, f): LabeledPoint(l, f))
            self.with_lp = True
        else:
            if self.data is not None:
                self.data = self.data.map(lambda lp: (lp.label, lp.features))
            self.with_lp = False

    def split_data(self, fractions, seed=123, balanced=False):
        """
        Particionamiento del conjunto de datos, en base a las proporciones dadas por *fractions*.
        Se hace mediante el uso de la función :func:`~learninspy.utils.data.split_data`.

        :param fractions: list de floats, indicando la fracción del total a tomar por cada dataset (deben sumar 1).
        :param seed: int, semilla a utilizar en el módulo *random* que hace el split.
        :param balanced: bool, si es *True* se recurre a la función :func:`~learninspy.utils.data.split_balanced`.
        :return: list de conjuntos :class:`learninspy.utils.data.DistributedLabeledDataSet`.
        """
        if balanced is True:
            sets = split_balanced(self.data, fractions, seed)
        else:
            sets = split_data(self.data, fractions, seed)
        sets = [DistributedLabeledDataSet(data) for data in sets]
        return sets

    def collect(self, unpersist=False):
        """
        Devuelve el conjunto de datos como lista, mediante la aplicación del método *collect()* sobre el RDD.

        :param unpersist: bool, indicando si además se quiere llamar al método *unpersist()* del RDD alojado.
        :return: list
        """
        data_list = self.data.collect()
        if unpersist is True:
            self.data.unpersist()
            self.data = None
        return data_list

    @property
    def shape(self):
        """
        Devuelve el tamaño del conjunto de datos alojado.

        :return: tuple, de cantidad de filas y columnas.
        """
        shape = None
        if self.data is not None:
            shape = self.rows, self.cols
        return shape


class LocalLabeledDataSet(LabeledDataSet):
    """
    Clase útil para manejar un conjunto etiquetado de datos. Dicho conjunto se almacena
    de manera local mediante una lista
    donde cada entrada posee un *pyspark.mllib.regression.LabeledPoint*.
    Se proveen funcionalidades para manejo de archivos, así como para partir el conjunto de datos
    (e.g. train, valid y test).

    :param data: list o numpy.ndarray o pyspark.rdd.RDD, o bien *None* si se desea iniciar un conjunto vacío.
    """
    @assert_features_label
    def __init__(self, data=None):
        self.with_lp = False  # Flag que indica si se almaceno entradas en forma de LabeledPoints
        if data is not None:
            if isinstance(data, pyspark.rdd.RDD):
                data = data.collect()
            if type(data[0]) is LabeledPoint:
                self.with_lp = True
        self.data = data
        if self.with_lp is False and self.data is not None:  # Por ahora que quede etiquetado con LabeledPoints
            self._labeled_point()

        if self.data is not None:
            self.rows = len(self.data)
            self.cols = len(self.features[0])

    @property
    def features(self):
        """
        Devuelve sólo las características del conjunto de datos, en el correspondiente orden almacenado.

        :return: list
        """
        if self.with_lp is True:
            features = map(lambda lp: lp.features, self.data)
        else:
            features = map(lambda (l, f): f, self.data)
        return features

    @property
    def labels(self):
        """
        Devuelve sólo las etiquetas del conjunto de datos, en el correspondiente orden almacenado.

        :return: list
        """
        if self.with_lp is True:
            labels = map(lambda lp: lp.label, self.data)
        else:
            labels = map(lambda (l, f): l, self.data)
        return labels

    def load_file(self, path, pos_label=-1):
        """
        Carga de conjunto de datos desde archivo. El formato aceptado es de archivos de texto, como CSV, donde
        los valores se separan por un caracter delimitador
        (configurable en :func:`~learninspy.utils.fileio.parse_point`).


        :param path: string, indicando la ruta de donde cargar los datos.
        :param pos_label: int, posición o n° de elemento de cada línea del archivo,
                          que corresponde al **label** (por defecto es -1, que corresponde a la última posición).
        """
        self.data = fileio.load_file_local(path, pos_label=pos_label)
        self._labeled_point()

        self.rows = len(self.data)
        self.cols = len(self.features[0])

    def save_file(self, path):
        """
        Guardar conjunto de datos en archivo de texto.

        .. warning:: No se encuentra implementada.

        :param path: string, indicando la ruta en donde se guardan los datos.
        """

        if self.with_lp is True:
            data = map(lambda lp: (lp.label, lp.features), self.data)
        else:
            data = self.data
        data = map(lambda (label, features): [label] + list(features), data)
        fileio.save_file_local(data, path)

    def _labeled_point(self):
        """
        Función para transformar cada entrada del conjunto de datos, de LabeledPoint a tupla o viceversa.

        """
        if self.with_lp is False:
            if self.data is not None:
                self.data = map(lambda (l, f): LabeledPoint(l, f), self.data)
            self.with_lp = True
        else:
            if self.data is not None:
                self.data = map(lambda lp: (lp.label, lp.features), self.data)
            self.with_lp = False

    def split_data(self, fractions, seed=123, balanced=False):
        """
        Particionamiento del conjunto de datos, en base a las proporciones dadas por *fractions*.
        Se hace mediante el uso de la función :func:`~learninspy.utils.data.split_data`.

        :param fractions: list de floats, indicando la fracción del total a tomar por cada dataset (deben sumar 1).
        :param seed: int, semilla a utilizar en el módulo *random* que hace el split.
        :param balanced: bool, si es *True* se recurre a la función :func:`~learninspy.utils.data.split_balanced`.
        :return: list de conjuntos :class:`learninspy.utils.data.LocalLabeledDataSet`.
        """
        if balanced is True:
            sets = split_balanced(self.data, fractions, seed)
        else:
            sets = split_data(self.data, fractions, seed)
        sets = [LocalLabeledDataSet(data) for data in sets]
        return sets

    def collect(self):
        """
        Función que retorna el conjunto de datos como una lista.
        Creada para lograr compatibilidad con DistributedLabeledDataSet.
        """
        return self.data

    @property
    def shape(self):
        """
        Devuelve el tamaño del conjunto de datos alojado.

        :return: tuple, de cantidad de filas y columnas.
        """
        shape = None
        if self.data is not None:
            shape = self.rows, self.cols
        return shape


# -- Funciones --

def label_to_vector(label, n_classes):
    """
    Función para mapear una etiqueta numérica a un vector de dimensión igual a **n_classes**,
    con todos sus elementos iguales a 0 excepto el de la posición **label**.

    :param label: int, pertenenciente al rango [0, *n_classes* - 1].
    :param n_classes: int, correspondiente a la cantidad de clases posibles para *label*.
    :return: numpy.ndarray
    """
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)


# TODO: hacer sample ponderando las clases por error en validacion
def subsample(data, size, balanced=True, seed=123):
    """
    Muestreo de data, con resultado balanceado por clases si se lo pide.

    :param data: list de LabeledPoint.
    :param size: int, tamaño del muestreo.
    :param seed: int, semilla del random.
    :return: list de LabeledPoint.

    """
    random.seed(seed)
    if balanced is True:  # Problema de clasificacion
        n_classes = int(max(map(lambda lp: lp.label, data))) + 1  # TODO: no siempre el batch va a tener todas las clases
        size /= n_classes  # es un int, y puede resultar menor al ingresado (se trunca)
        sample = []
        for c in xrange(n_classes):
            batch_class = filter(lambda lp: lp.label == c, data)  # Filtro entradas que pertenezcan a la clase c
            batch = random.sample(batch_class, size)
            sample.extend(batch)  # Agrego el batch al vector de muestreo
    else:  # Problema de regresion
        sample = random.sample(data, size)
    random.shuffle(sample)  # Mezclo para agregar aleatoriedad
    return sample


def split_data(data, fractions, seed=123):
    """
    Split data en sets en base a fractions.

    :param data: list o numpy.ndarray o pyspark.rdd.RDD.
    :param fractions: list de floats, indicando la fracción del total a tomar por cada dataset (deben sumar 1).
    :param seed: int, semilla a utilizar en el módulo *random* que hace el split.
    :return: list de conjuntos (e.g. train, valid, test)
    """
    # Verifico que fractions sea correcto
    assert (sum(fractions) <= 1.0, Exception("Fracciones para conjuntos incorrectas!"))
    if isinstance(data, pyspark.rdd.RDD):
        sets = data.randomSplit(fractions, seed)  # Uso la funcion del RDD. TODO: no esta devolviendo tamaños acordes a fractions
    else:
        # Mezclo un poco los datos
        random.seed(seed)
        random.shuffle(data)
        # Segmento conjuntos
        size_data = len(data)
        size_split = map(lambda f: int(round(size_data * f)), fractions)  # Int de Round para que no se pierdan rows
        index_split = [0]
        for s in size_split:
            index_split.append(index_split[-1]+s)
        sets = [data[i:f] for i, f in zip(index_split, index_split[1:])]
    return sets


def split_balanced(data, fractions, seed=123):
    """
    Split data en sets en base a fractions, pero de forma balanceada por clases (fracción aplicada a cada clase).

    .. note:: Se infiere la cantidad total de clases en base a los labels en 'data'.

    :param data: list o numpy.ndarray o pyspark.rdd.RDD.
    :param fractions: list de floats, indicando la fracción del total a tomar por cada dataset (deben sumar 1).
    :param seed: int, semilla a utilizar en el módulo *random* que hace el split.
    :return: list de conjuntos (e.g. train, valid, test)
    """

    # 'fractions' es aplicado a cada conjunto formado por c/ clase
    # 'n' indica la cantidad de clases
    # Verifico que fractions sea correcto
    assert (sum(fractions) <= 1.0, Exception("Fracciones para conjuntos incorrectas!"))
    if isinstance(data, pyspark.rdd.RDD):
        data = data.collect()  # Solución provisoria
    assert isinstance(data[0], LabeledPoint), Exception("Solo se puede operar con conjunto de LabeledPoints!")
    n = int(max(map(lambda lp: lp.label, data))) + 1 # Cantidad de clases
    sets_per_class = [filter(lambda lp: lp.label == c, data) for c in xrange(n)]  # Conjuntos de datos por cada clase
    sets_splitted = [split_data(s, fractions, seed) for s in sets_per_class]  # Split de cada conjunto de los anteriores
    sets = reduce(lambda l1, l2: map(add, l1, l2), sets_splitted)  # Sumo splits correspondientes por cada clase
    # Mezclo un poco los datos
    random.seed(seed)
    for i in xrange(len(sets)):
        random.shuffle(sets[i])
    return sets


def label_data(data, labels):
    """
    Función para etiquetar cada elemento de **data** con su correspondiente de **label**,
    formando una list de elementos
    `pyspark.mllib.regression.LabeledPoint
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LabeledPoint>`_.

    :param data: list o numpy.ndarray, correspondiente a **features**
    :param labels: list o numpy.ndarray, correspondiente a **labels**
    :return: list
    """
    # TODO: podría hacerse que label sea int, señalando la posicion de data donde esta la columna de labels
    labeled_data = map(lambda (x, y): LabeledPoint(y, x), zip(data, labels))
    return labeled_data


# -- Datos de ejemplo --

def load_ccpp(path=None):
    """
    Carga del conjunto de datos de Combined Cycle Power Plant extraido del
    UCI Machine Learning repository <http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant>`_.

    :param path: string, ruta al archivo 'ccpp.csv'.
    :return: list de LabeledPoints.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'examples/datasets/ccpp.csv')
    data = fileio.load_file_local(path)
    data = map(lambda (l, f): LabeledPoint(l, f), data)
    return data

def load_iris(path=None):
    """
    Carga del conjunto de datos de Iris.

    :param path: string, ruta al archivo 'iris.csv'.
    :return: list de LabeledPoints.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'examples/datasets/iris.csv')
    data = fileio.load_file_local(path)
    data = map(lambda (l, f): LabeledPoint(l, f), data)
    return data


def load_mnist(path=None):
    """
    Carga del conjunto de datos original de MNIST.

    :param path: string, ruta al archivo 'mnist.pkl.gz'.
    :return: tuple de lists con LabeledPoints, correspondientes a los conjuntos de train, valid y test respectivamente.
    """
    # Datos y procedim extraídos de http://deeplearning.net/tutorial/gettingstarted.html
    if path is None:
        path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'examples/datasets/mnist.pkl.gz')
    f = gzip.open(path, 'rb')
    train, valid, test = cPickle.load(f)
    f.close()

    # Etiqueto datos en LabeledPoints
    train = label_data(data=train[0], labels=train[1])
    valid = label_data(data=valid[0], labels=valid[1])
    test = label_data(data=test[0], labels=test[1])

    return train, valid, test
