__author__ = 'leferrad'

import dnn.model as mod
from dnn.optimization import OptimizerParameters
from dnn.stops import criterion
import time
from utils.data import split_data, label_data
from dnn.evaluation import ClassificationMetrics
from sklearn import datasets

net_params = mod.DeepLearningParams(units_layers=[4, 10, 5, 3], activation='Softplus',
                                    dropout_ratios=[0.5, 0.5, 0.0], classification=True)

local_criterions = [criterion['MaxIterations'](50),
                    criterion['AchieveTolerance'](0.99, key='hits')]

global_criterions = [criterion['MaxIterations'](5),
                     criterion['AchieveTolerance'](0.99, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', criterions=local_criterions)


neural_net = mod.NeuralNetwork(net_params)

print "Cargando base de datos ..."
data = datasets.load_iris()
features = data.data
labels = data.target
print "Size de la data: ", features.shape

train, valid, test = split_data(label_data(features, labels), [.7, .2, .1])

print "Entrenando red neuronal ..."
t1 = time.time()
hits_valid = neural_net.fit(train, valid, mini_batch=50, parallelism=4, criterions=global_criterions,
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





