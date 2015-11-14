__author__ = 'leferrad'

# Dependencias externas
import pyspark.rdd

# Dependencias internas
from learninspy.context import sc as sc
from learninspy.common.asserts import *


class DistributedNeurons(object):
    # Vectores son guardados como vectores columna
    def __init__(self, mat, shape):
        if isinstance(mat, pyspark.rdd.PipelinedRDD):
            self.matRDD = mat
        else:  # mat es un np.array
            if len(shape) == 1:
                shape += (1,)  # Le agrego la dimension que le falta
            if shape[1] != 1:
                mat = mat.T  # Guardo por columnas
                if shape[0] == 1:
                    shape = shape[::-1]  # Invierto para que sea vector columna
            self.matRDD = sc.parallelize(mat)

        self.rows = shape[0]
        self.cols = shape[1]

    # OPERADORES:

    def __mul__(self, other):
        return DistributedNeurons(self.matRDD.map(lambda x: x * other), self.shape)

    def __div__(self, other):
        return DistributedNeurons(self.matRDD.map(lambda x: x / other), self.shape)

    def __add__(self, other):
        rdd = other.matrix()
        return DistributedNeurons(self.matRDD.zip(rdd).map(lambda (x, y): x + y), self.shape)

    def __sub__(self, other):
        rdd = other.matrix()
        return DistributedNeurons(self.matRDD.zip(rdd).map(lambda (x, y): x - y), self.shape)

    def __pow__(self, power, modulo=None):
        return DistributedNeurons(self.matRDD.map(lambda x: x ** power), self.shape)

    # TRANSFORMACIONES:
    @assert_sametype
    @assert_samedimension
    def mul_elemwise(self, rdd):
        rdd = rdd.matrix()
        return DistributedNeurons(self.matRDD.zip(rdd).map(lambda (x, y): x * y), self.shape)

    @assert_sametype
    @assert_samedimension
    def sum_array(self, rdd):
        rdd = rdd.matrix()
        return DistributedNeurons(self.matRDD.zip(rdd).map(lambda (x, y): x + y), self.shape)

    def activation(self, fun):
        return DistributedNeurons(self.matRDD.map(lambda x: fun(x)), self.shape)

    @assert_sametype
    @assert_samedimension
    def mse_d(self, y):
        y = y.matrix()
        n = self.count()
        error = self.matRDD.zip(y).map(lambda (o, t): t - o)
        errorDiv = error.map(lambda val: val * (2.0 / n))
        return DistributedNeurons(errorDiv, self.shape)

    def dropout(self, p):
        # Aplicable para vectores unicamente
        index = np.random.rand(*self.shape) > p
        indexRDD = sc.parallelize(index)  # Vector RDD de bools, que indican que valor "tirar"
        # Notar que se escala el resultado diviendo por p
        # Retorno arreglo con dropout, y el vector con indices de las unidades "tiradas"
        zipped = self.matRDD.zip(indexRDD)
        dropped = zipped.map(lambda (x, y): (x * y) / p)
        return DistributedNeurons(dropped, self.shape), indexRDD

    def zip(self, rdd):
        return self.matRDD.zip(rdd)

    # ACCIONES:

    def mul_array(self, array): # Lento
        # Ver decoradores para chequear dimensiones validas
        if isinstance(array, np.ndarray):
            array = DistributedNeurons(array, array.shape)
        return self.mul_arrayrdd(array)

    def mul_array2(self,array):  #Uso operaciones de numpy
        if isinstance(array, DistributedNeurons) or isinstance(array, pyspark.rdd.PipelinedRDD):
            array = array.collect()
        shape = self.rows, array.shape[1]
        return DistributedNeurons(self.collect().dot(array), shape)

    @assert_sametype
    @assert_matchdimension
    def mul_arrayrdd(self, rdd):
        # Supongo que la matriz que multiplica a rdd esta transpuesta
        shape = self.rows, rdd.cols  # mul es un np.ndarray
        rdd = rdd.matrix()
        zipped = self.matRDD.zip(rdd)
        mapped = zipped.map(lambda (col, row): np.outer(col, row))
        mul = mapped.reduce(lambda x, y: x + y)
        if shape[1] == 1:  # Si es vector columna
            mul = mul.reshape(-1)  # lo redimensiono para que sea 1D
        return DistributedNeurons(mul, shape)

    @assert_sametype
    @assert_matchdimension
    def mul_arrayrdd2(self, rdd): # No anda bien pq devuelve un vector fila (en el reduce)
        # Supongo que la matriz que multiplica a rdd esta transpuesta
        shape = self.rows, rdd.cols  # mul es un np.ndarray
        rdd = rdd.matrix()
        zipped = self.matRDD.zip(rdd)
        mapped = zipped.map(lambda (col, row): (1, np.outer(col, row)))
        mul = mapped.reduceByKey(lambda x, y: x + y).map(lambda (k, v): v)
        if shape[1] == 1: # Si es vector columna, redimensiono para que sea 1D
            mul = mul.map(lambda col: col.reshape(-1))
        return DistributedNeurons(mul, shape)

    def outer(self, array):
        if isinstance(array, DistributedNeurons) or isinstance(array, pyspark.rdd.PipelinedRDD):
            array = array.collect()
        mat = self.collect()
        res = np.outer(mat, array)
        shape = res.shape
        return DistributedNeurons(res, shape)

    @assert_sametype
    @assert_samedimension
    def dot(self, rdd):
        rdd = rdd.matrix()
        return self.mul_elemwise(rdd).sum()

    def l1(self):
        gradient = self.matRDD.map(lambda x: np.sign(x))
        cost = self.matRDD.map(lambda x: abs(x).sum()).sum()
        return cost, DistributedNeurons(gradient, self.shape)

    def l2(self):
        gradient = self.matRDD
        cost = self.matRDD.map(lambda x: (x ** 2).sum()).sum()
        return cost, DistributedNeurons(gradient, self.shape)

    def loss(self, fun, y):
        y = DistributedNeurons(y, y.shape)
        return self.mse(y)

    def loss_d(self, fun, y):
        y = DistributedNeurons(y, y.shape)
        return self.mse_d(y)

    @assert_sametype
    @assert_samedimension
    def mse(self, y):
        y = y.matrix()
        n = self.count()
        error = self.matRDD.zip(y).map(lambda (o, t): (t - o) ** 2)
        sumError = error.sum()
        return sumError / (1.0 * n)

    def softmax(self):
        # Uso de tip de implementacion (http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression)
        maxmat = self.matRDD.max()
        matminus = self.matRDD.map(lambda x: x - maxmat)
        exp = np.exp
        matexp = matminus.map(lambda o: exp(o))
        sumexp = matexp.sum()
        softmat = matexp.map(lambda o: o / sumexp)
        return DistributedNeurons(softmat, self.shape)

    def collect(self):
        return np.array(self.matRDD.collect()).T  # Se transpone para retornar por filas

    def sum(self):
        return self.matRDD.sum()

    # MISC

    def persist(self):
        self.matRDD.cache()
        return

    def unpersist(self):
        self.matRDD.unpersist()
        return

    @property
    def shape(self):
        return self.rows, self.cols

    def count(self):
        rows, cols = self.shape
        return rows * cols

    def matrix(self):
        return self.matRDD


#----------------------------

class LocalNeurons(object):
    # Vectores son guardados como vectores columna
    def __init__(self, mat, shape):

        if isinstance(mat, pyspark.rdd.PipelinedRDD):
            mat = mat.collect()
        if len(shape) == 1:
            shape += (1,)  # Le agrego la dimension que le falta
        if isinstance(mat, list):
            mat = np.array(mat)
        self.matrix = mat.reshape(shape)
        self.rows = shape[0]
        self.cols = shape[1]

    # OPERADORES:

    def __mul__(self, other):
        if isinstance(other, LocalNeurons):
            other = other.matrix
        return LocalNeurons(self.matrix * other, self.shape)

    def __div__(self, other):
        if isinstance(other, LocalNeurons):
            other = other.matrix
        return LocalNeurons(self.matrix / other, self.shape)

    def __sub__(self, other):
        if isinstance(other, LocalNeurons):
            other = other.matrix
        return LocalNeurons(self.matrix - other, self.shape)

    def __add__(self, other):
        if isinstance(other, LocalNeurons):
            other = other.matrix
        return LocalNeurons(self.matrix + other, self.shape)

    def __pow__(self, power, modulo=None):
        return LocalNeurons(self.matrix ** power, self.shape)

    def __sizeof__(self):
        return self.matrix.nbytes + 88

    # TRANSFORMACIONES:
    #@assert_samedimension
    def mul_elemwise(self, array):
        if isinstance(array, LocalNeurons):
            array = array.matrix
        return LocalNeurons(np.multiply(self.matrix, array), self.shape)

    #@assert_samedimension
    def sum_array(self, array):
        if isinstance(array, LocalNeurons):
            array = array.matrix
        return LocalNeurons(self.matrix + array, self.shape)

    def activation(self, fun):
        return LocalNeurons(map(lambda x: fun(x), self.matrix), self.shape)

    #@assert_samedimension
    def mse_d(self, y):
        n = self.count()
        error = y - self.matrix
        errorDiv = error * (2.0 / n)
        return LocalNeurons(errorDiv, self.shape)

    def dropout(self, p):
        # Aplicable para vectores unicamente
        index = np.random.rand(*self.shape) > p  # Vector de bools, que indican que valor "tirar"
        # Notar que se escala el resultado diviendo por p
        # Retorno arreglo con dropout, y el vector con indices de las unidades "tiradas"
        dropped = self.matrix * index / p
        return LocalNeurons(dropped, self.shape), index

    def zip(self, rdd):
        pass

    # ACCIONES:

    #@assert_matchdimension
    def mul_array(self, array):
        if isinstance(array, LocalNeurons):
            array = array.matrix
        arrshape = array.shape
        if len(arrshape) == 1:
            arrshape += (1,)  # Le agrego la dimension que le falta
        shape = self.rows, arrshape[1]
        return LocalNeurons(self.matrix.dot(array), shape)

    def mul_array2(self, array):
        pass

    def mul_arrayrdd(self, rdd):
        pass

    def outer(self, array):
        if isinstance(array, LocalNeurons):
            array = array.matrix
        res = np.outer(self.matrix, array)
        shape = res.shape
        return LocalNeurons(res, shape)

    #@assert_samedimension
    def dot(self, vec):
        return self.matrix.dot(vec)

    def l1(self):
        cost = sum(sum(abs(self.matrix)))
        gradient = np.sign(self.matrix)  # en x=0 no es diferenciable, pero que sea igual a 0 la derivada anda bien
        return cost, LocalNeurons(gradient, self.shape)

    def l2(self):
        cost = sum(sum(self.matrix ** 2))
        gradient = self.matrix
        return cost, LocalNeurons(gradient, self.shape)

    def loss(self, fun, y):
        return fun(self.matrix, y)

    def loss_d(self, fun, y):
        return LocalNeurons(fun(self.matrix, y), self.shape)

    #@assert_samedimension
    def mse(self, y):
        n = self.count()
        error = (y - self.matrix) ** 2
        sumError = sum(error)
        return sumError / (1.0 * n)

    def softmax(self):
        # Uso de tip de implementacion (http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression)
        x = self.matrix
        x = x - max(x)  # Se previene valores muy grandes del exp con valores altos de x
        softmat = np.exp(x) / (sum(np.exp(x)))
        return LocalNeurons(softmat, self.shape)

    def transpose(self):
        return LocalNeurons(self.matrix.transpose(), self.shape[::-1])

    def collect(self):
        return self.matrix

    def sum(self):
        return sum(self.matrix)

    # MISC

    @property
    def shape(self):
        return self.rows, self.cols

    def count(self):
        rows, cols = self.shape
        return rows * cols

    def persist(self):
        pass

    def unpersist(self):
        pass


