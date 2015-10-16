__author__ = 'leferrad'


import numpy as np
import pyspark.rdd
from pyspark.mllib.regression import LabeledPoint


class PCA(object):
    # Ver explicacion en http://cs231n.github.io/neural-networks-2/
    def __init__(self, x):
        if isinstance(x,pyspark.rdd.PipelinedRDD):
            x = x.collect()
        self.x = x
        if isinstance(x[0], LabeledPoint):
            x = np.array(map(lambda lp: lp.features, x))
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
        if isinstance(data[0], LabeledPoint):
            label = map(lambda lp: lp.label, data)  # Guardo labels para concatenarlos al final
            data = np.array(map(lambda lp: lp.features, data))
            lp_data = True
        data -= self.mean  # zero-center sobre data (importante)
        if standarize is True:
            data /= self.std
        xrot = np.dot(data, self.u[:, :k])
        if whitening is True:
            xrot = xrot / np.sqrt(self.s[:k] + self.whitening_offset)
        if lp_data is True:
            xrot = map(lambda (f, l): LabeledPoint(l, f), zip(xrot.tolist(), label))
        return xrot




