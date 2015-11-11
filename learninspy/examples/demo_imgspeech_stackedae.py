__author__ = 'leferrad'

import time

from learninspy import dnn as mod
from learninspy.dnn.autoencoder import StackedAutoencoder
from learninspy.dnn.optimization import OptimizerParameters
from learninspy.dnn.stops import criterion
from learninspy.utils.data import LabeledDataSet, StandardScaler
from learninspy.dnn.evaluation import ClassificationMetrics
from learninspy.context import sc
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

# Uso clase hecha para manejo de DataSet (almacena en RDD)
dataset = LabeledDataSet(zip(labels, features))
train, valid, test = dataset.split_data([.7, .2, .1])  # Particiono conjuntos

# -----
# Aplico PCA
pca = PCA(train)
train = pca.transform()
valid = pca.transform(data=valid)
test = pca.transform(data=test)
k = pca.k

# -----
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

net_params = mod.DeepLearningParams(units_layers=[k, 25, 10, 3], activation='ReLU',
                                    dropout_ratios=[0.5, 0.5, 0.0], classification=True, seed=seed)

local_stops = [criterion['MaxIterations'](50),
                    criterion['AchieveTolerance'](0.90, key='hits')]

ae_stops = [criterion['Patience'](8, grow_offset=0.5),
                     criterion['AchieveTolerance'](0.95, key='hits')]

ft_stops = [criterion['MaxIterations'](20),
                     criterion['AchieveTolerance'](0.95, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, merge_criter='log_avg')

print "Entrenando stacked autoencoder ..."
t1 = time.time()
sae = StackedAutoencoder(net_params)
hits_valid = sae.fit(train, valid, mini_batch=100, parallelism=4,
                     stops=ae_stops, optimizer_params=opt_params)

print "Ajuste fino ..."
hits_valid = sae.finetune(train, valid, mini_batch=100, parallelism=4, criterions=ft_stops,
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





