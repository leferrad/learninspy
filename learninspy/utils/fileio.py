#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Librerias internas
from learninspy.context import sc

# Librerias de Python
import re
import logging
import string


def parse_point(line, delimiter=r'[ ,|;"]+'):
    """
    Convierte un string en list, separando elementos mediante la aparición un caracter delimitador entre ellos.

    :param line: string, contenedor de los caracteres que se desea separar.
    :param delimiter: string, contenedor de los posibles caracteres delimitadores.
    :return: list con elementos deseados.
    """
    # TODO dar posibilidad de cambiar delimiter
    values = [float(x) for x in re.split(delimiter, line)]
    return values


def label_point(row, pos_label=-1):
    label = row[pos_label]
    row.pop(pos_label)
    features = row
    return label, features


# Adaptación de http://code.activestate.com/recipes/173220/
def is_text_file(path):
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


# Loader
def load_file(path, pos_label=-1, delimiter=r'[ ,|;"]+'):
    if is_text_file(path):
        dataset = (sc.textFile(path)
                     .map(lambda p: parse_point(p, delimiter))
                     .map(lambda row: label_point(row, pos_label))
                   )
    else:
        dataset = sc.binaryFiles(path)  # TODO: mejorar esto
    return dataset


# Saver
# TODO mejorar esto que ni anda
def save_file(data, path):
    if is_text_file(path):
        data.saveAsTextFile(path+'.txt')
    else:
        data.saveAsPickleFile(path)
    return


def get_logger(name='learninspy', level=logging.INFO):
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