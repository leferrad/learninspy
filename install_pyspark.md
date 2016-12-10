Quick guide to setup PySpark in a Ubuntu/Debian distribution.
=============================================================

## Dependencies
* Java SE Development Kit
* Spark [>=1.3.x, 2.0.x]
* Python 2.6 or higher.

##### JAVA installation
Spark requires Java 7+, which you can download from the following distributions:
* ORACLE
    ```
    $ sudo apt-add-repository ppa:webupd8team/java
    $ sudo apt-get update
    $ sudo apt-get install oracle-java7-installer
    ```
* Open JDK
    ```
    $ sudo apt-get install openjdk-7-jre openjdk-7-jdk
    ```

To check if Java has been installed correctly, run 
```
$ java -version
```

##### Apache Spark installation
The following code will create a folder in ``/usr/local`` for allocating Spark's binaries (i.e. pre-built distribution of Spark, which is recommended for a quick installation).
```
$ wget http://d3kbcqa49mib13.cloudfront.net/spark-1.6.0-bin-hadoop2.6.tgz
$ tar xvf spark-1.6.0-bin-hadoop2.6.tgz
$ mv spark-1.6.0-bin-hadoop2.6.tgz /usr/local/spark-1.6.0
```
Now you need to configure some environment variables. Therefore open the ``~/.bashrc`` file and add the following lines:

```
export SPARK_HOME=/usr/local/spark
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/python/build:$PYTHONPATH
# Configure py4j allocated on Spark directory
PYFORJ=`ls -1 $SPARK_HOME/python/lib/py4j-*-src.zip | head -1`
export PYTHONPATH=$PYTHONPATH:$PYFORJ
# OPTIONAL: Configure JAVA 7 OpenJDK
export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
export PYSPARK_DRIVER_PYTHON=ipython  # it can be 'ipython' or 'jupyter-notebook'
```

In these lines, you are setting up the root home of Spark and including it into the Python path. Also, it is configured the PY4J package included in Spark directory, which is responsible for translating between Scala and Python. Finally, in case of having installed JAVA 7 OpenJDK, it is configured its path (otherwise, you need to point ``JAVA_HOME`` to its real path). Additionally, it is configured a variable for indicating how it should be launched PySpark (in this case, with IPython shell).

Now, you can check if it's all running by these ways:
1) Importing 'pyspark' package from Python shell:
    ```python
    import pyspark
    # It should be imported with no errors
    ```
2) Launching 'pyspark' script:
    ```
    $ cd /usr/local/spark-1.6.0
    $ ./bin/pyspark
    # Spark should be initialized correctly...
    ```
