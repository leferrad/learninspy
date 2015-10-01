__author__ = 'leferrad'

from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.rdd
from context import sc

class Metrics(MulticlassMetrics):

    def __init__(self, prediction_label):
        if not isinstance(prediction_label, pyspark.rdd.PipelinedRDD):
            prediction_label = sc.parallelize(prediction_label)
        MulticlassMetrics.__init__(self, prediction_label)


