__author__ = 'leferrad'

# Librerias de Python
import time
import os

# Librerias internas
import learninspy.dnn.model as mod
from learninspy.dnn.optimization import OptimizerParameters
from learninspy.dnn.stops import criterion
from learninspy.utils.data import StandardScaler, LabeledDataSet
from learninspy.dnn.evaluation import ClassificationMetrics


net_params = mod.DeepLearningParams(units_layers=[4, 10, 5, 3], activation='Softplus',
                                    dropout_ratios=[0.5, 0.5, 0.0], classification=True)

local_stops = [criterion['MaxIterations'](50),
               criterion['AchieveTolerance'](0.95, key='hits')]

global_stops = [criterion['MaxIterations'](5),
                criterion['AchieveTolerance'](0.99, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops)


neural_net = mod.NeuralNetwork(net_params)

print "Cargando base de datos ..."
# data = datasets.load_iris()
# features = data.data
# labels = data.target
dataset = LabeledDataSet()
dataset.load_file(os.curdir+'/datasets/iris.csv')
print "Size de la data: "
print dataset.shape

print "Creando conjuntos de train, valid y test ..."
train, valid, test = dataset.split_data([.7, .2, .1])  # Particiono conjuntos
# Standarize data
std = StandardScaler()
std.fit(train)
train = std.transform(train)
valid = std.transform(valid)
test = std.transform(test)
# Collect de RDD en list
train = train.collect()
valid = valid.collect()
test = test.collect()

print "Entrenando red neuronal ..."
t1 = time.time()
hits_valid = neural_net.fit(train, valid, mini_batch=50, parallelism=4, stops=global_stops,
                            optimizer_params=opt_params)
hits_test, predict = neural_net.evaluate(test, predictions=True)
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits_test

print "Metricas: "
labels = map(lambda lp: float(lp.label), test)
metrics = ClassificationMetrics(zip(predict, labels), 3)
print "Precision: ", metrics.precision()
print "Recall: ", metrics.recall()
print "Confusion: "
print metrics.confusion_matrix()





