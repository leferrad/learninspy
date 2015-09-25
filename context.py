__author__ = 'leferrad'

import os
import sys

# Variables de entorno
#os.environ['SPARK_HOME'] = "/usr/local/spark/"
#os.environ['LEARNINSPY_HOME'] = os.path.realpath('')

# Append pyspark  to Python Path
#sys.path.append("/usr/local/spark/python/")
#sys.path.append("/usr/local/spark/python/build")  # Esta soluciona el problema con py4j

from pyspark import SparkContext, SparkConf
#from utils.util import LearninspyLogger

if 'sc' not in locals() or sc is None:
    appName = 'demo1'
    master = 'local[*]'
    conf = (SparkConf().setAppName(appName)
            .set("Xmx", "2g")
            .setMaster(master)
            .set("spark.logConf", "false"))
    sc = SparkContext(conf=conf)

#if 'logger' not in locals():
#    logger = LearninspyLogger('INFO')
