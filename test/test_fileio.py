#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.utils.fileio import *
from learninspy.context import sc

import os
import shutil

logger = get_logger(name=__name__)

TEMP_PATH = "/tmp/"


def test_parse_point():
    line = "0,1,2,3,4,5,6,7,8,9"
    parsed = parse_point(line)
    assert map(lambda n: int(n), parsed) == range(10)


def test_is_text_file():
    # Test con archivo de texto y binario
    bin_filename = TEMP_PATH + "bin_file.bin"
    txt_filename = TEMP_PATH + "txt_file.txt"

    with open(bin_filename, "wb") as f:
        f.write(os.urandom(1024))

    with open(txt_filename, "w") as f:
        f.write("probando 123")

    assert is_text_file(bin_filename) is False
    assert is_text_file(txt_filename) is True


def test_file_spark():
    logger.info("Testeo de manejo de archivos de texto con Spark...")
    filename = TEMP_PATH + "spark_data.txt"
    if os.path.exists(filename):
        shutil.rmtree(filename)

    rdd_data = sc.range(10)  # dummy data

    save_file_spark(rdd_data, filename)  # Saving
    data = load_file_spark(filename, pos_label=0)  # Loading
    data = data.map(lambda (l, f): int(l))  # Remain only int labels of dataset

    assert rdd_data.collect() == sorted(data.collect())


def test_file_local():
    logger.info("Testeo de manejo de archivos de texto en forma local...")
    filename = TEMP_PATH + "local_data.txt"
    if os.path.exists(filename):
        os.remove(filename)

    local_data = zip(range(10), range(10, 20))  # dummy data

    save_file_local(local_data, filename)  # Saving
    data = load_file_local(filename, pos_label=0)  # Loading
    data = map(lambda (l, f): (int(l), int(f[0])), data)  # Remain int labels and values of features on dataset

    assert local_data == data