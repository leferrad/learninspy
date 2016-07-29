__author__ = 'leferrad'

# Librerias de Python
import time
import os

# Librerias internas
from learninspy.core import model as mod
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import StandardScaler, LocalLabeledDataSet
from learninspy.utils.evaluation import ClassificationMetrics


net_params = mod.NetworkParameters(units_layers=[4, 10, 10, 3], activation='Softplus',
                                   dropout_ratios=[0.5, 0.5, 0.0], classification=True)

local_stops = [criterion['MaxIterations'](10),
               criterion['AchieveTolerance'](0.95, key='hits')]

global_stops = [criterion['MaxIterations'](30),
                criterion['AchieveTolerance'](0.95, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, merge_criter='w_avg')


neural_net = mod.NeuralNetwork(net_params)

print "Cargando base de datos ..."
dataset = LocalLabeledDataSet()
dataset.load_file(os.path.dirname(os.path.realpath(__file__))+'/datasets/iris.csv')
print "Size de la data: "
print dataset.shape

print "Creando conjuntos de train, valid y test ..."
train, valid, test = dataset.split_data([.5, .3, .2])  # Particiono conjuntos
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
hits_valid = neural_net.fit(train, valid, valid_iters=1, mini_batch=30, parallelism=4, stops=global_stops,
                            optimizer_params=opt_params, keep_best=True, reproducible=True)
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





