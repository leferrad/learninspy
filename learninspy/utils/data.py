#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np
import pyspark.rdd
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler as StdSc

# Dependencias internas
import learninspy.utils.fileio as fileio
from learninspy.context import sc
from learninspy.utils.asserts import assert_features_label

# Librerias de Python
import random


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

        :param dataset: pyspark.rdd.RDD o numpy.array o :class:`.LabeledDataSet`

        """
        if type(dataset) is LabeledDataSet:
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

        :param dataset: pyspark.rdd.RDD o numpy.array o :class:`.LabeledDataSet`

        """
        labels = None  # Por si el dataset viene con labels
        if type(dataset) is LabeledDataSet:
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
            dataset = LabeledDataSet(labels.zip(dataset))
        return dataset


class LabeledDataSet(object):
    """
    Clase útil para manejar un conjunto etiquetado de datos. Dicho conjunto se almacena
    como un `pyspark.rdd.RDD <https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD>`_
    donde cada entrada posee una lista de **features** y su correspondiente **label**.
    Se proveen funcionalidades para manejo de archivos, así como para partir el conjunto de datos
    (e.g. train, valid y test).

    :param data: list o numpy.array o pyspark.rdd.RDD, o bien *None* si se desea iniciar un conjunto vacío.
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
        self.data = fileio.load_file(path, pos_label=pos_label)
        self._labeled_point()

    def save_file(self, path):  # TODO mejorar pq no anda
        """
        Guardar conjunto de datos en archivo de texto.


        :param path: string, indicando la ruta en donde se guardan los datos.
        """

        fileio.save_file(self.data, path)

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

    def split_data(self, fractions, seed=123):
        """
        Particionamiento del conjunto de datos, en base a las proporciones dadas por *fractions*.
        Se hace mediante el uso de la función :func:`~learninspy.utils.data.split_data`.
        """
        sets = split_data(self.data, fractions, seed)
        sets = [LabeledDataSet(data) for data in sets]
        return sets

    def collect(self, unpersist=True):
        """
        Devuelve el conjunto de datos como lista, mediante la aplicación del método *collect()* sobre un
        `pyspark.rdd.RDD <https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD>`_.

        :param unpersist: bool, indicando si además se quiere llamar al método *unpersist()* del **pyspark.rdd.RDD** alojado.
        :return: list

        """
        data_list = self.data.collect()
        if unpersist is True:
            self.data.unpersist()
        return data_list

    @property
    def shape(self):
        """
        Devuelve el tamaño del conjunto de datos alojado.

        :return: tuple, de cantidad de filas y columnas.
        """
        shape = None
        if self.data is not None:
            rows = self.data.count()
            cols = len(self.features.take(1)[0].toArray())
            shape = (rows, cols)
        return shape


def label_to_vector(label, n_classes):
    """
    Función para mapear una etiqueta numérica a un vector de dimensión igual a **n_classes**,
    con todos sus elementos iguales a 0 excepto el de la posición **label**.

    :param label: int, pertenenciente al rango [0, *n_classes* - 1].
    :param n_classes: int, correspondiente a la cantidad de clases posibles para *label*.
    :return: numpy.array
    """
    lab = np.zeros((n_classes, 1), dtype=np.int8)
    label = int(label)
    lab[label] = 1
    return np.array(lab)


# TODO: hacer sample ponderando las clases por error en validacion
def subsample(data, size, balanced=True, seed=123):
    """
    Muestreo de data, con resultado balanceado por clases si se lo pide.

    :param data: list of LabeledPoint
    :param size: int, tamaño del muestreo.
    :param seed: int, semilla del random.
    :return: list

    """
    random.seed(seed)
    if balanced is True:  #Problema de clasificacion
        n_classes = int(max(map(lambda lp: lp.label, data))) + 1
        size /= n_classes  # es un int, y puede resultar menor al ingresado (se trunca)
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
    Split data en sets en base a fractions.

    :param data: list o np.array
    :param fractions: list [f_train, f_valid, f_test]
    :param seed: int, semilla para el random
    :return: list de conjuntos (e.g. train, valid, test)
    """
    # Verifico que fractions sea correcto
    # TODO: assert (sum(fractions) <= 1.0, Exception("Fracciones para conjuntos incorrectas!"))
    if isinstance(data, pyspark.rdd.RDD):
        sets = data.randomSplit(fractions, seed)  # Uso la funcion del RDD
    else:
        # Mezclo un poco los datos
        random.seed(seed)
        random.shuffle(data)
        # Segmento conjuntos
        size_data = len(data)
        size_split = map(lambda f: int(size_data * f), fractions)
        index_split = [0] + size_split[:-1]
        sets = [data[i:i+size] for i, size in zip(index_split, size_split)]
    return sets


def label_data(data, label):
    """
    Función para etiquetar cada elemento de **data** con su correspondiente de **label**,
    formando una list de elementos
    `pyspark.mllib.regression.LabeledPoint
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LabeledPoint>`_.

    :param data: list o numpy.array, correspondiente a **features**
    :param label: list o numpy.array, correspondiente a **labels**
    :return: list
    """
    labeled_data = map(lambda (x, y): LabeledPoint(y, x), zip(data, label))
    return labeled_data

