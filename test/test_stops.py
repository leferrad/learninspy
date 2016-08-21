#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

from learninspy.core.stops import *
from learninspy.utils.fileio import get_logger

from functools import partial

logger = get_logger(name=__name__)


def test_max_iterations():
    max_iter = 10
    stop = MaxIterations(max_iter=max_iter)

    results = {'iterations': 5}
    assert stop(results) is False

    results = {'iterations': 10}
    assert stop(results) is True


def test_achieve_tolerance():
    tol = 1e-3
    key = 'cost'
    stop = AchieveTolerance(tolerance=tol, key=key)

    results = {key: 0.5}
    assert stop(results) is False

    results = {key: 1e-8}
    assert stop(results) is True

    tol = 0.9
    key = 'hits'
    stop = AchieveTolerance(tolerance=tol, key=key)

    results = {key: 0.5}
    assert stop(results) is False

    results = {key: 1.0}
    assert stop(results) is True


def test_modulo_niter():
    n = 5
    stop = ModuloNIterations(n)

    results = {'iterations': 8}
    assert stop(results) is False

    results = {'iterations': 15}
    assert stop(results) is True


def test_time_elapsed():
    seconds = 5
    stop = TimeElapsed(sec=seconds)

    results = {}
    time.sleep(2)
    assert stop(results) is False

    time.sleep(3)
    assert stop(results) is True


def test_notbetterthan_after():
    minimal = 0.7
    after = 10
    key = 'hits'
    stop = NotBetterThanAfter(minimal, after, key=key)

    results = {'iterations': 15, key: 0.8}
    assert stop(results) is False
    results = {'iterations': 15, key: 0.5}
    assert stop(results) is True

    key = 'cost'
    stop = NotBetterThanAfter(minimal, after, key=key)
    results = {'iterations': 15, key: 0.3}
    assert stop(results) is False
    results = {'iterations': 15, key: 1.5}
    assert stop(results) is True


def test_patience_increase():
    # Adaptacion de https://github.com/BRML/climin/blob/master/test/test_stops.py
    func_hits = partial(next, iter([0.2, 0.2, 0.2, 0.5, 0.2, 0.2, 0.2]))

    initial = 3
    key = 'hits'
    grow_factor = 2
    threshold = 0.05

    stop = Patience(initial, key, grow_factor=grow_factor, threshold=threshold)

    assert not stop({'iterations': 0, key: func_hits()})
    assert not stop({'iterations': 1, key: func_hits()})
    assert not stop({'iterations': 2, key: func_hits()})
    assert not stop({'iterations': 3, key: func_hits()})
    assert not stop({'iterations': 4, key: func_hits()})
    assert not stop({'iterations': 5, key: func_hits()})
    assert stop({'iterations': 6, key: func_hits()})

    func_cost = partial(next, iter([0.2, 0.2, 0.2, 1e-4, 0.2, 0.2, 0.2]))

    key = 'cost'
    grow_factor = 2
    threshold = 0.05

    stop = Patience(initial, key, grow_factor=grow_factor, threshold=threshold)

    assert not stop({'iterations': 0, key: func_cost()})
    assert not stop({'iterations': 1, key: func_cost()})
    assert not stop({'iterations': 2, key: func_cost()})
    assert not stop({'iterations': 3, key: func_cost()})
    assert not stop({'iterations': 4, key: func_cost()})
    assert not stop({'iterations': 5, key: func_cost()})
    assert stop({'iterations': 6, key: func_cost()})


def test_on_signal():
    stop = OnSignal()

    assert stop(None) is False
    # Simular Ctrl+C
    stop._handler(signal.SIGINT, None)
    assert stop(None) is True