#! /usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages

setup(
    name="learninspy",
    version="0.1.0",
    author="Leandro Ferrado",
    author_email="ljferrado@gmail.com",
    url="https://github.com/leferrad/learninspy",
    packages=find_packages(exclude=['examples', 'docs', 'test']),
    license="LICENSE",
    description="Framework of deep learning for PySpark",
    long_description=open("README.md").read(),
    install_requires=open("requirements.txt").read().split()
)
