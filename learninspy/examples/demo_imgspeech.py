__author__ = 'leferrad'

import time

from learninspy.dnn.model import DeepLearningParams, NeuralNetwork
from learninspy.dnn.optimization import OptimizerParameters
from learninspy.dnn.stops import criterion
from learninspy.utils.data import StandardScaler, LabeledDataSet
from learninspy.utils.evaluation import ClassificationMetrics
from learninspy.utils.feature import PCA

print "Cargando base de datos de entrenamiento..."
# Uso clase hecha para manejo de DataSet (almacena en RDD)
train = LabeledDataSet()
train.load_file("/media/leeandro04/Data/Downloads/ImaginedSpeechDATA/Speech Imagery Task_mat/prj_mat/S1_train.dat")
rows = train.data.count()
cols = len(train.features.take(1)[0].toArray())
print "Size: ", rows, " x ", cols

train, valid = train.split_data([.8, .2])  # Particiono conjuntos

print "Cargando base de datos de testeo..."
test = LabeledDataSet()
test.load_file("/media/leeandro04/Data/Downloads/ImaginedSpeechDATA/Speech Imagery Task_mat/prj_mat/S1_test.dat")
rows = test.data.count()
cols = len(test.features.take(1)[0].toArray())
print "Size: ", rows, " x ", cols
"""
# Aplico PCA
pca = PCA(train)
train = pca.transform()
valid = pca.transform(data=valid)
test = pca.transform(data=test)
k = pca.k
print "Componentes principales tomadas: ", k
"""
# Standarize data
std = StandardScaler()
std.fit(train)
train = std.transform(train)
valid = std.transform(valid)
test = std.transform(test)

# Seleccion de parametros para la construccion de red neuronal
net_params = DeepLearningParams(units_layers=[512, 400, 200, 100, 3], activation='Softplus',
                                dropout_ratios=[0.2, 0.2, 0.2, 0.0], classification=True)
neural_net = NeuralNetwork(net_params)

# Seleccion de parametros de optimizacion
local_stops = [criterion['MaxIterations'](10),
               criterion['AchieveTolerance'](0.90, key='hits')]

global_stops = [criterion['MaxIterations'](100),
                criterion['AchieveTolerance'](0.95, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, merge_criter='log_avg')

print "Entrenando red neuronal ..."
t1 = time.time()
hits_valid = neural_net.fit(train, valid, mini_batch=30, parallelism=4, stops=global_stops,
                            optimizer_params=opt_params, keep_best=True)
t1f = time.time() - t1

# Resultados
test = test.collect()
hits_test, predict = neural_net.evaluate(test, predictions=True)
print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits_test

print "Metricas: "
labels = map(lambda lp: float(lp.label), test)
metrics = ClassificationMetrics(zip(predict, labels), 3)
print "Precision: ", metrics.precision()
print "Recall: ", metrics.recall()
print "Confusion: "
print metrics.confusion_matrix()





