#!/usr/bin/env python
# -*- coding: utf-8 -*-7

# Librerias de Python
import signal
import sys
import time


class MaxIterations(object):

    def __init__(self, max_iter):
        self.max_iter = max_iter

    def __call__(self, results):
        return results['iterations'] >= self.max_iter

    def __str__(self):
        return "Stop at a maximum of "+str(self.max_iter)+" iterations."


class AchieveTolerance(object):

    def __init__(self, tolerance, key='hits'):
        self.tolerance = tolerance
        self.key = key

    def __call__(self, results):
        return results[self.key] >= self.tolerance

    def __str__(self):
        return "Stop when a tolerance of "+str(self.tolerance)+" is achieved in "+self.key+"."


class ModuloNIterations(object):

    def __init__(self, n):
        self.n = n

    def __call__(self, results):
        return results['iterations'] % self.n == 0

    def __str__(self):
        return "Stop when an iteration is modulo of "+str(self.n)+"."


class TimeElapsed(object):

    def __init__(self, sec):
        self.sec = sec
        self.start = time.time()

    def __call__(self, results):
        return time.time() - self.start > self.sec

    def __str__(self):
        return "Stop when "+str(self.sec)+" seconds have elapsed."


class NotBetterThanAfter(object):

    def __init__(self, minimal, after, key='hits'):
        self.minimal = minimal
        self.after = after
        self.key = key

    def __call__(self, info):
        return info['iterations'] > self.after and info[self.key] < self.minimal

    def __str__(self):
        return "Stop when "+self.key+" does not improve a minimal of " + \
               str(self.minimal)+" after "+str(self.after)+" iterations."

# TODO: Hace un NoImprovementAfter que cada N iteraciones tome un máximo y un mínimo y compruebe que se mejoró en X porciento


class Patience(object):
    """

    """
    def __init__(self, initial, key='hits', grow_factor=1., grow_offset=0.,
                 threshold=0.05):
        if grow_factor == 1 and grow_offset == 0:
            raise ValueError('need to specify either grow_factor != 1'
                             'or grow_offset != 0')
        self.key = key
        self.patience = initial
        self.grow_factor = grow_factor
        self.grow_offset = grow_offset
        self.threshold = threshold

        self.best_value = float('inf')
        if self.key == 'hits':
            self.best_value = -self.best_value  # Se busca maximizar key

    def __call__(self, results):
        i = results['iterations']
        value = results[self.key]
        if self.key == 'hits':
            # Se busca maximizar key
            better_value = value > self.best_value

        else:
            # Se busca minimizar key (que es 'cost')
            better_value = value < self.best_value
        if better_value is True:
            if (value - self.best_value) > self.threshold and i > 0:
                self.patience = max(i * self.grow_factor + self.grow_offset,
                                    self.patience)
            self.best_value = value
        return i >= self.patience


# TODO: incoportar posibilidad de admitir Ctrl+c sin perder todo el trabajo
class OnUnixSignal(object):
    """Stopping criterion that is sensitive to some signal."""

    def __init__(self, sig=signal.SIGINT):
        """Return a stopping criterion that stops upon a signal.
        Previous handler will be overwritten.
        Parameters
        ----------
        sig : signal, optional [default: signal.SIGINT]
            Signal upon which to stop.
        """
        self.sig = sig
        self.stopped = False
        self._register()

    def _register(self):
        self.prev_handler = signal.signal(self.sig, self.handler)

    def handler(self, signal, frame):
        self.stopped = True

    def __call__(self, info):
        res, self.stopped = self.stopped, False
        return res

    def __del__(self):
        signal.signal(self.sig, self.prev_handler)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._register()


class OnWindowsSignal(object):
    """Stopping criterion that is sensitive to signals Ctrl-C or Ctrl-Break
    on Windows."""

    def __init__(self, sig=None):
        """Return a stopping criterion that stops upon a signal.
        Previous handlers will be overwritten.
        Parameters
        ----------
        sig : signal, optional [default: [0,1]]
            Signal upon which to stop.
            Default encodes signal.SIGINT and signal.SIGBREAK.
        """
        self.sig = [0, 1] if sig is None else sig
        self.stopped = False
        self._register()

    def _register(self):
        import win32api
        win32api.SetConsoleCtrlHandler(self.handler, 1)

    def handler(self, ctrl_type):
        if ctrl_type in self.sig:  # Ctrl-C and Ctrl-Break
            self.stopped = True
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler

    def __call__(self, info):
        res, self.stopped = self.stopped, False
        return res

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._register()


OnSignal = OnWindowsSignal if sys.platform == 'win32' else OnUnixSignal

criterion = {'MaxIterations': MaxIterations, 'AchieveTolerance': AchieveTolerance,
                  'ModuloNIterations': ModuloNIterations, 'TimeElapsed': TimeElapsed,
                  'NotBetterThanAfter': NotBetterThanAfter, 'Patience': Patience,
                  'OnSignal': OnSignal}
