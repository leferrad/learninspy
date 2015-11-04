__author__ = 'leferrad'


import numpy as np
import pyspark.rdd
from pyspark.mllib.regression import LabeledPoint
from utils.data import LabeledDataSet


class PCA(object):
    # Ver explicacion en http://cs231n.github.io/neural-networks-2/
    def __init__(self, x):
        self.x = x
        if type(x) is LabeledDataSet:
            x = x.features.collect()
        x = np.array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.whitening_offset = 1e-5

        # Se computa la matriz de covarianza
        cov = np.dot(x.T, x) / x.shape[0]

        # SVD factorizacion de la matriz de covarianza
        u, s, v = np.linalg.svd(cov)

        # Columnas de U son los eigenvectores (ordenados por sus eigenvalores)
        # S contiene los valores singulares (eigenvalores al cuadrado)
        self.u = u
        self.v = v
        self.s = s

    def transform(self, k=None, data=None, standarize=True, whitening=True):
        if k is None:
            k = int(self.x.shape[1] * .85)  # retengo 85% de la varianza
        if data is None:
            data = self.x
        lp_data = False  # Flag que indica que data es LabeledPoint, y debo preservar sus labels
        label = None
        if type(data) is LabeledDataSet:
            label = data.labels.collect()  # Guardo labels para concatenarlos al final
            data = np.array(data.features.collect())
            lp_data = True
        data -= self.mean  # zero-center sobre data (importante)
        if standarize is True:
            data /= self.std
        xrot = np.dot(data, self.u[:, :k])
        if whitening is True:
            xrot = xrot / np.sqrt(self.s[:k] + self.whitening_offset)
        if lp_data is True:
            xrot = LabeledDataSet(zip(label, xrot.tolist()))
        return xrot




