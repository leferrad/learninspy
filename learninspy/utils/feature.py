#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Módulo destinado a funcionalidades para realizar extracción de características sobre un conjunto de datos."""

__author__ = 'leferrad'

from learninspy.utils.data import DistributedLabeledDataSet, LabeledDataSet, LocalLabeledDataSet

import numpy as np


class PCA(object):
    """
    Clase utilizada para aplicar *análisis de componentes principales* o *PCA*
    sobre un conjunto de datos, con lo cual se proyecta cada uno de sus puntos o
    vectores en un espacio de menor dimensión.

    Ver más info en Wikipedia: `Principal component analysis <https://en.wikipedia.org/wiki/Principal_component_analysis>`_.

    :param x: list de lists, o instancia de :class:`~learninspy.utils.data.LabeledDataSet`.
    :param threshold_k: float, umbral de varianza máxima a retener sobre los datoss
     en caso de que no se indique la dimensión *k* final.

    >>> from learninspy.utils.data import load_iris
    >>> data = load_iris()
    >>> features = map(lambda lp: lp.features, data)
    >>> pca = PCA(features, threshold_k=0.99)
    >>> pca.k  # K óptima determinada por la varianza cubierta el threshold_k
    2
    >>> transformed = pca.transform(k=3)
    >>> print len(transformed[0])
    3

    """
    # Ver explicacion en http://cs231n.github.io/neural-networks-2/
    # TODO: Ver si conviene usar como parametro 'sigmas' que se sumen al mean, en lugar de una varianza acumulada
    def __init__(self, x, threshold_k=0.95):
        self.x = x
        type_x = type(x)
        if type_x is DistributedLabeledDataSet:
            x = x.features.collect()
        elif type_x is LocalLabeledDataSet:
            x = x.features
        x = np.array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=1)
        self.whitening_offset = 1e-5  # TODO: ver si conviene tenerlo como parámetro, aunque no creo
        self.k = None
        # Umbral de varianza explicada, para sacar un k optimo
        self.threshold_k = threshold_k

        # Se computa la matriz de covarianza
        cov = np.dot(x.T, x) / x.shape[0]

        # SVD factorizacion de la matriz de covarianza
        u, s, v = np.linalg.svd(cov)

        # Columnas de U son los eigenvectores (ordenados por sus eigenvalores)
        # S contiene los valores singulares (eigenvalores al cuadrado)
        self.u = u
        self.v = v
        self.s = s
        self.k = self._optimal_k()  # Defino una k optima por defecto

    def transform(self, k=None, data=None, standarize=False, whitening=True):
        """
        Transformación de datos mediante PCA hacia una dimensión *k*.

        :param k: int, si es *None* se utiliza el *k* óptimo calculado por la clase.
        :param data: numpy.ndarray, o instancia de :class:`~learninspy.utils.data.LabeledDataSet`.
         Si es *None*, se aplica sobre los datos utilizados inicialmente para el ajuste.
        :param standarize: bool, si es *True* se dividen los datos por su desvío estándar (std).
        :param whitening: bool, si es *True* se aplica el proceso de whitening
         (ver más información en `Wikipedia <https://en.wikipedia.org/wiki/Whitening_transformation>`_).
        :return: numpy.ndarray, con los vectores transformados.
        """
        if k is not None:
            self.k = k
        if data is None:
            data = self.x
        label = None
        type_data = type(data)  # Para saber si la entrada es un dataset, y debo conservar sus labels
        if type_data is DistributedLabeledDataSet:
            label = data.labels.collect()  # Guardo labels para concatenarlos al final
            data = np.array(data.features.collect())
        elif type_data is LocalLabeledDataSet:
            label = data.labels  # Guardo labels para concatenarlos al final
            data = np.array(data.features)
        data -= self.mean  # zero-center sobre data (importante)
        if standarize is True:
            data /= self.std
        xrot = np.dot(data, self.u[:, :self.k])
        if whitening is True:
            xrot = xrot / np.sqrt(self.s[:self.k] + self.whitening_offset)
        if type_data is DistributedLabeledDataSet or type_data is LocalLabeledDataSet:
            xrot = type_data(zip(label, xrot.tolist()))
        return xrot

    def _optimal_k(self):
        """
        Barrido de k hasta cubrir un threshold de varianza dado por self.threshold_k (e.g 95%)
        """
        var_total = sum(self.s)
        opt_k = 1
        for k in xrange(1, len(self.s)):
            explained_var = sum(self.s[:k]) / var_total
            if explained_var >= self.threshold_k:
                opt_k = k
                break
        return opt_k



