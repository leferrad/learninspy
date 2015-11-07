__author__ = 'leferrad'

import dnn.model as mod
from dnn.autoencoder import StackedAutoencoder
from dnn.optimization import OptimizerParameters
from dnn.stops import criterion
import time
from utils.data import split_data, label_data
from dnn.evaluation import ClassificationMetrics
from sklearn import datasets

net_params = mod.DeepLearningParams(units_layers=[4, 4, 3], activation='Softplus',
                                    dropout_ratios=[0.5, 0.0], classification=True)
sae_params = mod.DeepLearningParams(units_layers=[4, 4, 3], activation='Tanh')

local_stops = [criterion['MaxIterations'](50),
                    criterion['AchieveTolerance'](0.99, key='hits')]

global_stops = [criterion['MaxIterations'](10),
                     criterion['AchieveTolerance'](0.99, key='hits')]

print "Cargando base de datos ..."
data = datasets.load_iris()
features = data.data
labels = data.target
print "Size de la data: ", features.shape

train, valid, test = split_data(label_data(features, labels), [.7, .2, .1])

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
                          optimizer_params=opt_params)
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