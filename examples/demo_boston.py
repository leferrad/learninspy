__author__ = 'leferrad'

import time

from sklearn import datasets

from learninspy.core.model import NeuralNetwork, NetworkParameters
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import split_data, label_data, LocalLabeledDataSet
from learninspy.utils.evaluation import RegressionMetrics


net_params = NetworkParameters(units_layers=[13, 8, 1], activation='Identity',
                               dropout_ratios=[0.0, 0.0], classification=False)

local_stops = [criterion['MaxIterations'](50),
               criterion['AchieveTolerance'](0.99, key='hits')]

global_stops = [criterion['MaxIterations'](30),
                criterion['AchieveTolerance'](0.99, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops)


neural_net = NeuralNetwork(net_params)

print "Cargando base de datos ..."
data = datasets.load_boston()
features = data.data
labels = data.target
data = LocalLabeledDataSet(zip(labels, features))
print "Size de la data: ", data.shape

train, valid, test = data.split_data([.7, .2, .1])

print "Entrenando red neuronal ..."
t1 = time.time()
hits_valid = neural_net.fit(train, valid, mini_batch=50, parallelism=4, stops=global_stops,
                            optimizer_params=opt_params, keep_best=True)
hits_test, predict = neural_net.evaluate(test, predictions=True)
t1f = time.time() - t1

print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits_test

print "Metricas: "
labels = map(lambda lp: float(lp.label), test)
metrics = RegressionMetrics(zip(predict, labels))
print "MSE: ", metrics.mse()
print "RMSE: ", metrics.rmse()
print "MAE: ", metrics.mae()
print "R-cuadrado: ", metrics.r2()
print "Explained Variance: ", metrics.explained_variance()
print zip(predict, labels)




