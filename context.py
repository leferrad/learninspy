__author__ = 'leferrad'

import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME'] = "/usr/local/spark/"

# Append pyspark  to Python Path
sys.path.append("/usr/local/spark/python/")
sys.path.append("/usr/local/spark/python/build")  # Esta soluciona el problema con py4j

from pyspark import SparkContext, SparkConf

if 'sc' not in locals() or sc is None:
    appName = "demo1"
    master = "local[*]"
    conf = (SparkConf().setAppName(appName)
            .setMaster(master)
            .set("Xmx", "5g")
            .set("spark.logConf", "false"))
    sc = SparkContext(conf=conf)

