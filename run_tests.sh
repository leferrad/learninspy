#!/usr/bin/env bash

if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:../
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python/lib/py4j-0.9-src.zip

export PYTHONWARNINGS="ignore"

# nosetests $@ --verbosity 2
python test/test_metrics.py
python test/test_autoencoder.py
