#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo destinado al tratamiento de archivos y parseo de datos,
y además se provee el logger utilizado por Learninspy en su ejecución.
"""

__author__ = 'leferrad'

from learninspy.context import sc

import re
import logging
import csv
import string


def parse_point(line, delimiter=r'[ ,|;"]+'):
    """
    Convierte un string en list, separando elementos mediante la aparición un caracter delimitador entre ellos.

    :param line: string, contenedor de los caracteres que se desea separar.
    :param delimiter: string, donde se indican los posibles caracteres delimitadores.
    :return: list con elementos deseados.
    """
    values = [float(x) for x in re.split(delimiter, line)]
    return values


# Adaptación de http://code.activestate.com/recipes/173220/
def is_text_file(path):
    """
    Función utilizada para reconocer si un archivo es probablemente de texto o del tipo binario.

    :param path: string, path al archivo a analizar.
    :return: bool. *True* si es un archivo de texto, *False* si es binario.
    """
    text_characters = "".join(map(chr, range(32, 127)) + list("\n\r\t\b"))
    _null_trans = string.maketrans("", "")
    s = open(path).read(512)
    condition = True
    if not s:  # Archivos vacíos se consideran como texto
        condition = True
    if "\0" in s:  # Posiblemente binario si tiene caracteres nulos
        condition = False
    # Obtener los caracteres que no son texto (i.e. mapear un caracter
    # a si mismo. para usar la opción 'remove' que desheche lo que es texto)
    nt = s.translate(_null_trans, text_characters)
    # If more than 30% non-text characters, then
    # this is considered a binary file
    # Si más del 30% son caracteres que no son texto,
    # entonces se considera binario al archivo
    if float(len(nt))/float(len(s)) > 0.30:
        condition = False
    return condition


def _label_point(row, pos_label=-1):
    """
    Función interna para dividir una línea en *label* y *features*.

    :param row: list
    :param pos_label: int, posición donde se ubica el *label*. Si es -1, se indica la última posición de la lista.
    :return: tuple de (label, features)
    """
    label = row[pos_label]
    row.pop(pos_label)
    features = row
    return label, features


# Loader

def load_file_spark(path, pos_label=-1, delimiter=r'[ ,|;"]+'):
    """
    Carga de un archivo de datos mediante Apache Spark en RDD.

    :param path: string, path al archivo.
    :param pos_label: int, posición donde se ubica el *label* para cada línea. Si es -1, se indica la última posición.
    :param delimiter: string, donde se indican los posibles caracteres delimitadores.
    :return: *pyspark.rdd.RDD* de LabeledPoints.
    """
    #assert is_text_file(path), ValueError("Sólo se aceptan archivos de texto!")
    dataset = (sc.textFile(path)
                 .map(lambda p: parse_point(p, delimiter))
                 .map(lambda row: _label_point(row, pos_label))
               )
    return dataset


def load_file_local(path, pos_label=-1):
    """
    Carga de un archivo de datos en forma local.

    :param path: string, path al archivo.
    :param pos_label: int, posición donde se ubica el *label* para cada línea. Si es -1, se indica la última posición.
    :return: list de LabeledPoints.
    """
    assert is_text_file(path), ValueError("Sólo se aceptan archivos de texto!")
    with open(path, 'rb') as f:
        # Uso de Sniffer para deducir el formato del archivo CSV
        dialect = csv.Sniffer().sniff(f.read(1024))  # Ver https://docs.python.org/3/library/csv.html
        f.seek(0)
        reader = csv.reader(f, dialect)
        dataset = [x for x in reader]
        dataset = map(lambda row: _label_point(row, pos_label), dataset)
    return dataset


# Saver

def save_file_spark(rdd_data, path):
    """
    Guarda el contenido de un RDD en un archivo de texto.

    :param rdd_data: *pyspark.rdd.RDD* de lists.
    :param path: string, indicando la ruta en donde se guarda el archivo.
    """
    rdd_data.saveAsTextFile(path)
    return


def save_file_local(data, path, delimiter=','):
    """
    Guardar el contenido de un arreglo de listas en un archivo de texto.

    :param data: list de lists
    :param path: string, indicando la ruta en donde se guarda el archivo.
    """
    with open(path, "w") as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in data:
            writer.writerow(line)
    return

def get_logger(name='learninspy', level=logging.INFO):
    """
    Función para obtener el logger de Learninspy.

    :param name: string
    :param level: instancias del *logging* de Python (e.g. logging.INFO, logging.DEBUG)
    :return: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger