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
        self.std = np.std(x, axis=0, ddof=1)
        self.whitening_offset = 1e-5
        self.k = None

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
        if k is not None:
            self.k = k
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
        xrot = np.dot(data, self.u[:, :self.k])
        if whitening is True:
            xrot = xrot / np.sqrt(self.s[:self.k] + self.whitening_offset)
        if lp_data is True:
            xrot = LabeledDataSet(zip(label, xrot.tolist()))
        return xrot

    def _optimal_k(self, threshold=0.85):
        # Barrido de k hasta cubrir un threshold de varianza (e.g 85%)
        var_total = sum(self.s)
        k_opt = 1
        for k in xrange(1, len(self.s)):
            var_explained = sum(self.s[:k]) / var_total
            if var_explained >= threshold:
                k_opt = k
                break
        return k_opt



