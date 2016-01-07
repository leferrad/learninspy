#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
from scipy.io import loadmat, savemat

# Librerias internas
from learninspy.context import sc

# Librerias de Python
import re

text_extensions = ['.dat', '.txt', '.csv']


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


# Checks
def is_text_file(path):
    return any([path.lower().endswith(ext) for ext in text_extensions])


def is_mat_file(path):
    return path.lower().endswith('.mat')


# Loader TODO mejorar manejo de .mat
def load_file(path, pos_label=-1, varname=None):
    if is_text_file(path):
        dataset = sc.textFile(path).map(parse_point).map(lambda row: label_point(row, pos_label))
    elif is_mat_file(path):
        dataset = loadmat(path)[varname]
    else:
        dataset = sc.binaryFiles(path).map(parse_point)
    return dataset


# Saver TODO mejorar esto que ni anda
def save_file(data, path):
    if is_text_file(path):
        data.saveAsTextFile(path+'.txt')
    elif is_mat_file(path):
        savemat(path, {'dataset': data.collect()})
    else:
        data.saveAsPickleFile(path)
    return
