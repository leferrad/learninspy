__author__ = 'leferrad'


import numpy as np
import pyspark.rdd
from sklearn.decomposition import PCA


class PCA_Whitening:
    trained = False
    # Ver explicacion en http://cs231n.github.io/neural-networks-2/
    def __call__(self, data, num_comp=None, whiten=True):
        if isinstance(data,pyspark.rdd.PipelinedRDD):
            # data debe ser un np.array()
            data = np.array(data.collect())
        self.data = data
        if num_comp is None:
            num_comp = data.shape[1] / 2  # Por defecto, que se reduzca a la mitad de dimensiones
        self.num_comp = num_comp
        self.whiten = whiten
        # Notar que se debe entrenar solo con el conjunto de entrenamiento (fit) y luego se aplica transformacion
        # al conjunto de validacion/test (Para no hacer "trampa" usando las propiedades estadisticas en la prediccion
        self.pca = PCA(n_components=num_comp, whiten=whiten)

    def train(self):
        data_pca = self.pca.fit_transform(self.data)
        self.trained = True
        return data_pca

    def test(self,data):
        assert self.trained is True, 'PCA no entrenado!'
        if isinstance(data,pyspark.rdd.PipelinedRDD):
            # data debe ser un np.array()
            data = np.array(data.collect())
        return self.pca.transform(data)