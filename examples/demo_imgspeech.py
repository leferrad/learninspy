__author__ = 'leferrad'

import dnn.model as mod
from dnn.optimization import OptimizerParameters
from dnn.stops import criterion
import time
from utils.data import split_data, label_data
from dnn.evaluation import ClassificationMetrics
from context import sc
from utils.feature import PCA

def parsePoint(line):
    values = [float(x) for x in line.split(';')]
    return values[-1], values[0:-1]
# -----
seed = 123
print "Cargando base de datos ..."
data = (sc.textFile("/home/leeandro04/Documentos/Datos/EEG/ImaginedSpeech-Brainliner/alldatalabel_cat3_Norm.dat")
        .map(parsePoint))
features = data.map(lambda (l,f): f).collect()
labels = data.map(lambda (l,f): l).collect()
print "Size de la data: ", len(features), " x ", len(features[0])

train, valid, test = split_data(label_data(features, labels), [.7, .2, .1], seed=seed)
# -----
# Aplico PCA
k = 80
pca = PCA(train)
train = pca.transform(k=k)
valid = pca.transform(k=k, data=valid)
test = pca.transform(k=k, data=test)

# -----
net_params = mod.DeepLearningParams(units_layers=[k, 100, 50, 3], activation='Softplus',
                                        dropout_ratios=[0.5, 0.5, 0.0], classification=True, seed=seed)
# TODO: que sea un dict, donde se agrupan en 'any' y en 'all' los criterios
local_criterions = [criterion['MaxIterations'](10),
                    criterion['AchieveTolerance'](0.99, key='hits')]

global_criterions = [criterion['Patience'](100, key='hits', grow_factor=1., grow_offset=5.,
                                           threshold=0.02),
                     criterion['AchieveTolerance'](0.99, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', criterions=local_criterions)

neural_net = mod.NeuralNetwork(net_params)
# -----
print "Entrenando red neuronal ..."
t1 = time.time()
hits_valid = neural_net.fit(train, valid, mini_batch=100, parallelism=4, criterions=global_criterions,
                            optimizer_params=opt_params, keep_best=True)
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





