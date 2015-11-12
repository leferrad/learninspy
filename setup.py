from setuptools import setup, find_packages

setup(
    name="learninspy",

    version="0.1.0",

    author="Leandro Ferrado",
    author_email="ljferrado@gmail.com",
    url="https://github.com/leferrad/learninspy",

    include_package_data=True,

    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    scripts=[],

    license="LICENSE",

    description="Framework of deep learning for PySpark",
    long_description=open("README.md").read(),

    install_requires=open("requeriments.txt").read().split()
)
