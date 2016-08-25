#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.utils.data import *
from learninspy.utils.fileio import get_logger
from learninspy.context import sc

import os
import shutil

logger = get_logger(name=__name__)

TEMP_PATH = "/tmp/"


class TestLabeledDataset(object):
    def __init__(self):
        logger.info("Testeo de instances de LabeledDataset con datos de MNIST")
        # Datos
        logger.info("Cargando datos en memoria...")
        train, valid, test = load_mnist()
        self.data = train[:50]

    def test_load_data(self):
        # DistributedLabeledDataSet
        logger.info("Testeando carga de datos en DistributedLabeledDataset...")
        distributed_dataset = DistributedLabeledDataSet(self.data)
        d_features = distributed_dataset.features
        d_labels = distributed_dataset.labels
        distributed_dataset._labeled_point()
        assert d_features.collect() == distributed_dataset.features.collect()
        assert d_labels.collect() == distributed_dataset.labels.collect()
        d_shape = distributed_dataset.shape
        distributed_dataset._labeled_point()

        # LocalLabeledDataset
        logger.info("Testeando carga de datos en LocalLabeledDataset...")
        local_dataset = LocalLabeledDataSet(distributed_dataset.data)
        l_features = local_dataset.features
        l_labels = local_dataset.labels
        local_dataset._labeled_point()
        assert l_features == local_dataset.features
        assert l_labels == local_dataset.labels
        l_shape = local_dataset.shape
        local_dataset._labeled_point()

        assert d_shape == l_shape

        distributed_dataset.collect(unpersist=True)

    @staticmethod
    def test_file_distributed():
        logger.info("Testeo de manejo de archivos de texto con DistributedLabeledDataSet...")
        filename = TEMP_PATH + "spark_data.txt"
        if os.path.exists(filename):
            shutil.rmtree(filename)

        rdd_data = sc.parallelize(zip(range(10), range(10, 20)))  # dummy data
        rdd_data = rdd_data.map(lambda (l, f): (l, [f]))
        distributed_data = DistributedLabeledDataSet(rdd_data)
        distributed_data.save_file(filename)  # Saving
        dataset = DistributedLabeledDataSet()
        dataset.load_file(filename, pos_label=0)  # Loading

        # LabeledPoints to tuple(l,f)
        distributed_data._labeled_point()
        dataset._labeled_point()

        res1 = sorted(distributed_data.collect(unpersist=True), key=lambda (l, f): l)
        res2 = sorted(dataset.collect(unpersist=True), key=lambda (l, f): l)
        assert res1 == res2

    @staticmethod
    def test_file_local():
        logger.info("Testeo de manejo de archivos de texto con LocalLabeledDataSet...")
        filename = TEMP_PATH + "local_data.txt"
        if os.path.exists(filename):
            os.remove(filename)

        data = zip(range(10), range(10, 20))  # dummy data
        data = map(lambda (l, f): (l, [f]), data)
        local_data = LocalLabeledDataSet(data)
        local_data.save_file(filename)  # Saving
        dataset = LocalLabeledDataSet()
        dataset.load_file(filename, pos_label=0)  # Loading

        # LabeledPoints to tuple(l,f)
        local_data._labeled_point()
        dataset._labeled_point()

        res1 = sorted(local_data.collect(), key=lambda (l, f): l)
        res2 = sorted(dataset.collect(), key=lambda (l, f): l)
        assert res1 == res2

    def test_split_data(self, seed=123):
        # DistributedLabeledDataSet
        logger.info("Testeo de split en DistributedLabeledDataset...")
        distributed_dataset = DistributedLabeledDataSet(self.data)
        set1, set2 = distributed_dataset.split_data([0.1, 0.9], seed=seed)
        assert set1.rows < distributed_dataset.rows

        set1, set2 = distributed_dataset.split_data([0.9, 0.1], balanced=True, seed=seed)
        assert set(set1.labels.collect()) == set(distributed_dataset.labels.collect())  # Contienen todas las clases

        # LocalLabeledDataSet
        logger.info("Testeo de split en LocalLabeledDataset...")
        local_dataset = LocalLabeledDataSet(self.data)
        set1, set2 = local_dataset.split_data([0.1, 0.9], seed=seed)
        assert set1.rows == 5

        set1, set2 = local_dataset.split_data([0.9, 0.1], balanced=True, seed=seed)
        assert set(set1.labels) == set(local_dataset.labels)  # Contienen todas las clases