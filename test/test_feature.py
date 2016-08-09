#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.utils.feature import PCA
from learninspy.utils.data import load_iris, LocalLabeledDataSet, DistributedLabeledDataSet, StandardScaler
from learninspy.utils.fileio import get_logger
import numpy as np

logger = get_logger(name=__name__)


def test_pca():
    logger.info("Testeando PCA...")
    data = load_iris()

    # Testeo de funcionalidades
    features = map(lambda lp: lp.features, data)
    pca = PCA(features, threshold_k=0.99)
    assert pca.k == 2
    transformed = pca.transform(k=3, standarize=True, whitening=True)
    assert len(transformed[0]) == 3

    # Testeo de soporte para DataSets
    local_data = LocalLabeledDataSet(data)
    pca_loc = PCA(local_data.features)
    distributed_data = DistributedLabeledDataSet(data)
    pca_dist = PCA(distributed_data.features.collect())
    assert np.array_equiv(pca_loc.transform(k=3, data=local_data).features,
                          pca_dist.transform(k=3, data=distributed_data).features.collect())


def test_stdscaler():
    logger.info("Testeando StandardScaler...")
    data = load_iris()

    # Testeo de funcionalidades
    features = map(lambda lp: lp.features, data)
    stdsc = StandardScaler(mean=True, std=True)
    stdsc.fit(features)
    transformed = stdsc.transform(features)
    assert np.isclose(np.mean(transformed), 0, rtol=1e-8)

    # Testeo de soporte para DataSets
    local_data = LocalLabeledDataSet(data)
    stdsc.fit(local_data.features)
    local_transformed = stdsc.transform(local_data)
    distributed_data = DistributedLabeledDataSet(data)
    stdsc.fit(distributed_data)
    distrib_transformed = stdsc.transform(distributed_data)
    assert np.allclose(local_transformed.features, distrib_transformed.features.collect())
