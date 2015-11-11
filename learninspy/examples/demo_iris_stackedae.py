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
from learninspy.dnn.autoencoder import StackedAutoencoder

units_layers = [4, 5, 10, 3]
net_params = mod.DeepLearningParams(units_layers=units_layers, activation='Softplus',
                                    dropout_ratios=[0.5, 0.5, 0.0], classification=True)
sae_params = mod.DeepLearningParams(units_layers=units_layers, activation='Tanh')

local_stops = [criterion['MaxIterations'](30),
                    criterion['AchieveTolerance'](0.95, key='hits')]

global_stops = [criterion['MaxIterations'](10),
                     criterion['AchieveTolerance'](0.99, key='hits')]

print "Cargando base de datos ..."
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

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops)

print "Entrenando stacked autoencoder ..."
t1 = time.time()
sae = StackedAutoencoder(net_params)
hits_valid = sae.fit(train, valid, mini_batch=20, parallelism=4,
                         stops=global_stops, optimizer_params=opt_params)
hits_test = sae.evaluate(test, predictions=False)
print "Tasa de aciertos de SAE en test: ", hits_test

print "Ajuste fino ..."
hits_valid = sae.finetune(train, valid, mini_batch=50, parallelism=4, criterions=global_stops,
                          optimizer_params=opt_params, keep_best=True)
hits_test, predict = sae.evaluate(test, predictions=True)
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits_test

print "Metricas: "
labels = map(lambda lp: float(lp.label), test)
metrics = ClassificationMetrics(zip(predict, labels), 3)
print "Precision: ", metrics.precision()
print "Recall: ", metrics.recall()
print "Confusion: "
print metrics.confusion_matrix()