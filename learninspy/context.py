__author__ = 'leferrad'

# Variables de entorno
#os.environ['SPARK_HOME'] = "/usr/local/spark/"
#os.environ['LEARNINSPY_HOME'] = os.path.realpath('')

# Append pyspark  to Python Path
#sys.path.append("/usr/local/spark/python/")
#sys.path.append("/usr/local/spark/python/build")  # Esta soluciona el problema con py4j

from pyspark import SparkContext, SparkConf
from utils.util import LearninspyLogger

#TODO: ver que hacer con esto
class LearninspyContext(object):
    def __init__(self, app_name='LearninspyApp', master='local', xmx='2g', log_conf='false'):
        self.app_name = app_name
        self.conf = (SparkConf().setAppName(self.app_name)
                     .set("Xmx", xmx).setMaster(master)
                     .set("spark.logConf", log_conf))
        self.sc = SparkContext(conf=self.conf)

if 'sc' not in locals() or sc is None:
    appName = 'demo1'
    #master = 'local[*]'
    extraJavaOptions = '-XX:+UseG1GC'
    conf = (SparkConf().setAppName(appName)
            .set("Xmx", "3g")
     #       .setMaster(master)
            .set('spark.driver.extraJavaOptions', extraJavaOptions)
            .set('spark.executor.extraJavaOptions', extraJavaOptions)
            .set("spark.storage.memoryFraction", "0.5")
            .set("spark.logConf", "false"))
    sc = SparkContext(conf=conf)
    #sc = LearninspyContext(master=master).sc

if 'logger' not in locals():
    logger = LearninspyLogger('INFO')

