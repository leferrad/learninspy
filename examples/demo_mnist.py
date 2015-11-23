__author__ = 'leferrad'

# Librerias de Python
import time
import os

# Librerias internas
from learninspy.core.model import NetworkParameters, NeuralNetwork
from learninspy.core.optimization import OptimizerParameters
from learninspy.core.stops import criterion
from learninspy.utils.data import StandardScaler, LabeledDataSet
from learninspy.utils.evaluation import ClassificationMetrics
from learninspy.utils.feature import PCA

# Uso clase hecha para manejo de DataSet (almacena en RDD)
train = LabeledDataSet()
train.load_file(os.path.dirname(os.path.realpath(__file__))+"/datasets/mnist_train_100.csv", pos_label=0)
rows = train.data.count()
cols = len(train.features.take(1)[0].toArray())
print "Size: ", rows, " x ", cols

train, valid = train.split_data([.8, .2])  # Particiono conjuntos

print "Cargando base de datos de testeo..."
test = LabeledDataSet()
test.load_file(os.path.dirname(os.path.realpath(__file__))+"/datasets/mnist_test_10.csv", pos_label=0)
rows = test.data.count()
cols = len(test.features.take(1)[0].toArray())
print "Size: ", rows, " x ", cols

# Aplico PCA
#pca = PCA(train)
#train = pca.transform()
#valid = pca.transform(data=valid)
#test = pca.transform(data=test)
#k = pca.k

# Standarize data
std = StandardScaler()
std.fit(train)
train = std.transform(train)
valid = std.transform(valid)
test = std.transform(test)

net_params = NetworkParameters(units_layers=[784, 300, 100, 10], activation='Softplus',
                                    strength_l2=1e-5, strength_l1=3e-5,
                                    dropout_ratios=[0.5, 0.5, 0.0], classification=True)

neural_net = NeuralNetwork(net_params)

local_stops = [criterion['MaxIterations'](10),
                    criterion['AchieveTolerance'](0.90, key='hits')]

global_stops = [criterion['MaxIterations'](50),
                     criterion['AchieveTolerance'](0.95, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, merge_criter='log_avg')

print "Entrenando red neuronal ..."
t1 = time.time()
hits_valid = neural_net.fit(train, valid, mini_batch=20, parallelism=4, stops=global_stops,
                            optimizer_params=opt_params, keep_best=True)
t1f = time.time() - t1

hits_test, predict = neural_net.evaluate(test, predictions=True)
print 'Tiempo: ', t1f, 'Tasa de acierto final: ', hits_test

print "Metricas: "
test = test.collect()
labels = map(lambda lp: float(lp.label), test)
metrics = ClassificationMetrics(zip(predict, labels), 10)
print "Precision: ", metrics.precision()
print "Recall: ", metrics.recall()
print "Confusion: "
print metrics.confusion_matrix()