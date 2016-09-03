#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script para configurar contexto en Spark."""

__author__ = 'leferrad'

from pyspark import SparkContext, SparkConf

import os

if 'sc' not in locals() or sc is None:
    appName = 'learninspy-app'
    if 'SPARK_MASTER_IP' not in os.environ.keys() and 'SPARK_MASTER_PORT' not in os.environ.keys():
        master = 'local[*]'  # default: local mode
    else:
        master = 'spark://'+os.environ['SPARK_MASTER_IP']+':'+os.environ['SPARK_MASTER_PORT']  # master defined
    extraJavaOptions = '-XX:+UseG1GC'
    conf = (SparkConf().setAppName(appName)
            .setMaster(master)
            .set('spark.ui.showConsoleProgress', False)  # Para que no muestre el progreso de los Stages (comentar sino)
            .set('spark.driver.extraJavaOptions', extraJavaOptions)
            .set('spark.executor.extraJavaOptions', extraJavaOptions)
            .set('spark.executor.extraJavaOptions', '-XX:+UseCompressedOops')  # Cuando se tiene menos de 32GB de RAM, punteros de 4 bytes en vez de 8 bytes
            .set("spark.storage.memoryFraction", "0.5")  # Deprecado en 2.0.0
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer", "256")
            .set("spark.rdd.compress", "true")
            .set("spark.logConf", "false"))
    sc = SparkContext.getOrCreate(conf=conf)

    from learninspy.utils.fileio import get_logger
    logger = get_logger(name=__name__)

    logger.info("Contexto de Spark inicializado.")

