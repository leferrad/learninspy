__author__ = 'leferrad'

import time

from learninspy.dnn.model import DeepLearningParams
from learninspy.dnn.autoencoder import StackedAutoencoder
from learninspy.dnn.optimization import OptimizerParameters
from learninspy.dnn.stops import criterion
from learninspy.utils.data import LabeledDataSet, StandardScaler
from learninspy.dnn.evaluation import ClassificationMetrics
from learninspy.context import sc


def parsePoint(line):
    values = [float(x) for x in line.split(';')]
    return values[-1], values[0:-1]
# -----
seed = 123
print "Cargando base de datos ..."
data = sc.textFile("/media/leeandro04/Data/Backup/P300/concatData/datalabels_cat5_FIRDec.dat").map(parsePoint)

features = data.map(lambda (l,f): f).collect()
labels = data.map(lambda (l,f): l).collect()
print "Size de la data: ", len(features), " x ", len(features[0])


# Uso clase hecha para manejo de DataSet (almacena en RDD)
dataset = LabeledDataSet(zip(labels, features))
train, valid, test = dataset.split_data([.7, .2, .1])  # Particiono conjuntos

# -----
# Aplico PCA
#pca = PCA(train)
#train = pca.transform()
#valid = pca.transform(data=valid)
#test = pca.transform(data=test)
#k = pca.k

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

units = [230, 100, 20, 2]
dropout_ae = [0.2, 0.2, 0.0]
dropout = [0.2, 0.5, 0.0]
net_params = DeepLearningParams(units_layers=units, activation='ReLU',
                                    dropout_ratios=dropout, classification=True, seed=seed)

local_stops = [criterion['MaxIterations'](10),
               criterion['AchieveTolerance'](0.90, key='hits')]

ae_stops = [criterion['MaxIterations'](10),
            criterion['AchieveTolerance'](0.95, key='hits')]

ft_stops = [criterion['MaxIterations'](20),
            criterion['AchieveTolerance'](0.95, key='hits')]

opt_params = OptimizerParameters(algorithm='Adadelta', stops=local_stops, merge_criter='log_avg')

print "Entrenando stacked autoencoder ..."
t1 = time.time()
sae = StackedAutoencoder(net_params, dropout=dropout_ae)
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
metrics = ClassificationMetrics(zip(predict, labels), 2)
print "Precision: ", metrics.precision()
print "Recall: ", metrics.recall()
print "Confusion: "
print metrics.confusion_matrix()





