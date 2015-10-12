__author__ = 'leeandro04'

import itertools
import signal
import sys
import time


class MaxIterations(object):

    def __init__(self, max_iter):
        self.max_iter = max_iter

    def __call__(self, results):
        return results['iterations'] >= self.max_iter


class AchieveTolerance(object):

    def __init__(self, tolerance, key='hits'):
        self.tolerance = tolerance
        self.key = key

    def __call__(self, results):
        return results[self.key] >= self.tolerance

class ModuloNIterations(object):

    def __init__(self, n):
        self.n = n

    def __call__(self, results):
        return results['iterations'] % self.n == 0


class TimeElapsed(object):

    def __init__(self, sec):
        self.sec = sec
        self.start = time.time()

    def __call__(self, results):
        return time.time() - self.start > self.sec


class NotBetterThanAfter(object):

    def __init__(self, minimal, after, key='hits'):
        self.minimal = minimal
        self.after = after
        self.key = key

    def __call__(self, info):
        return info['iterations'] > self.after and info[self.key] >= self.minimal


class IsNaN(object):

    def __init__(self, keys=None):
        if keys is None:
            keys = []
        self.keys = keys

    def __call__(self, results):
        #return any([isnan(results.get(key, 0)) for key in self.keys])
        pass


class Patience(object):
    """Stop criterion inspired by Bengio's patience method.
    The idea is to increase the number of iterations until stopping by
    a multiplicative and/or additive constant once a new best candidate is
    found.
    Attributes
    ----------
    func_or_key : function, hashable
        Either a function or a hashable object. In the first case, the function
        will be called to get the latest loss. In the second case, the loss
        will be obtained from the in the corresponding field of the ``info``
        dictionary.
    initial : int
        Initial patience. Lower bound on the number of iterations.
    grow_factor : float
        Everytime we find a sufficiently better candidate (determined by
        ``threshold``) we increase the patience multiplicatively by
        ``grow_factor``.
    grow_offset : float
        Everytime we find a sufficiently better candidate (determined by
        ``threshold``) we increase the patience additively by ``grow_offset``.
    threshold : float, optional, default: 1e-4
        A loss of a is assumed to be a better candidate than b, if a is larger
        than b by a margin of ``threshold``.
    """

    def __init__(self, initial, key='hits', grow_factor=1., grow_offset=0.,
                 threshold=1e-4):
        if grow_factor == 1 and grow_offset == 0:
            raise ValueError('need to specify either grow_factor != 1'
                             'or grow_offset != 0')
        self.key = key
        self.patience = initial
        self.grow_factor = grow_factor
        self.grow_offset = grow_offset
        self.threshold = threshold

        self.best_value = float('inf')

    def __call__(self, results):
        i = results['iterations']
        value = results[self.key]
        if value < self.best_value:
            if (self.best_value - value) > self.threshold and i > 0:
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
