from distutils.core import setup

setup(
    name="Learninspy",

    version="0.1.0",

    author="Leandro Ferrado",
    author_email="ljferrado@gmail.com",
    url="https://github.com/leferrad/learninspy",

    include_package_data=True,

    packages=["app"],
    scripts=[],

    license="LICENSE",

    description="Framework of deep learning for PySpark",
    long_description=open("README.md").read(),

    install_requires=open("requeriments.txt").read().split()
)
