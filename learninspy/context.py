__author__ = 'leferrad'

# Variables de entorno
#os.environ['SPARK_HOME'] = "/usr/local/spark/"
#os.environ['LEARNINSPY_HOME'] = os.path.realpath('')

# Append pyspark  to Python Path
#sys.path.append("/usr/local/spark/python/")
#sys.path.append("/usr/local/spark/python/build")  # Esta soluciona el problema con py4j

from pyspark import SparkContext, SparkConf

if 'sc' not in locals() or sc is None:
    appName = 'demo1'
    #master = 'local[*]'
    extraJavaOptions = '-XX:+UseG1GC'
    conf = (SparkConf().setAppName(appName)
            .set("Xmx", "3g")
    #       .setMaster(master)
            .set('spark.ui.showConsoleProgress', False)  # Para que no muestre el progreso de los Stages (comentar sino)
            .set('spark.driver.extraJavaOptions', extraJavaOptions)
            .set('spark.executor.extraJavaOptions', extraJavaOptions)
            .set('spark.executor.extraJavaOptions', '-XX:+UseCompressedOops')  # Cuando se tiene menos de 32GB de RAM, punteros de 4 bytes en vez de 8 bytes
    #       .set("spark.storage.memoryFraction", "0.5")
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")  # Tambien habria que incrementar spark.kryoserializer.buffer
            .set("spark.logConf", "false"))
    sc = SparkContext(conf=conf)

if 'logger' not in locals():
    from learninspy.utils.fileio import get_logger
    logger = get_logger(name=__name__)

