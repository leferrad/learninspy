__author__ = 'leferrad'

import numpy as np

class ClassificationMetrics(object):
    # Ver http://machine-learning.tumblr.com/post/1209400132/mathematical-definitions-for-precisionrecall-for
    def __init__(self, predicted_actual, n_classes):
        self.predicted_actual = predicted_actual
        self.tp = []
        self.fp = []
        self.fn = []
        for c in xrange(n_classes):
            self.tp.append(sum(map(lambda (p, a): p == c and a == c, predicted_actual)))
            self.fp.append(sum(map(lambda (p, a): p == c and a != c, predicted_actual)))
            self.fn.append(sum(map(lambda (p, a): p != c and a == c, predicted_actual)))
        self.n_classes = n_classes
        self.n_elem = len(predicted_actual)

    def accuracy(self, label=None):
        if label is None:
            acc = sum(map(lambda (pre, act): pre == act, self.predicted_actual)) / float(self.n_elem)
        else:
            acc = self.tp[label] / float(self.tp[label] + self.fp[label] + self.fn[label])
        return acc

    def precision(self, label=None, macro=False):
        if label is None:
            if macro is True:
                p = sum([self.precision(c) for c in xrange(self.n_classes)])
                p /= float(self.n_classes)
            else:
                p = sum(self.tp) / float(sum(map(lambda (tp, fp): tp + fp, zip(self.tp, self.fp))))
        else:
            if self.tp[label] == 0.0 and self.fp[label] == 0.0:
                p = 1.0
            else:
                p = self.tp[label] / float(self.tp[label] + self.fp[label])
        return p

    def recall(self, label=None, macro=False):
        if label is None:
            if macro is True:
                r = sum([self.recall(c) for c in xrange(self.n_classes)])
                r /= float(self.n_classes)
            else:
                r = sum(self.tp) / float(sum(map(lambda (tp, fn): tp + fn, zip(self.tp, self.fn))))
        else:
            if self.tp[label] == 0.0 and self.fn[label] == 0.0:
                r = 1.0
            else:
                r = self.tp[label] / float(self.tp[label] + self.fn[label])
        return r

    def f_measure(self, label=None, beta=None):
        if beta is None:
            beta = 1
        ppv = self.precision(label)
        tpr = self.recall(label)
        f_score = (1 + beta*beta)*(ppv * tpr) / (beta*beta*ppv + tpr)
        return f_score

    def confusion_matrix(self):
        conf_mat = []  # Matriz de confusion final
        for r in xrange(self.n_classes):
            pre_act = filter(lambda (p, a): a == r, self.predicted_actual)
            for c in xrange(self.n_classes):
                conf_mat.append(sum(map(lambda (p, a): p == c, pre_act)))
        return np.array(conf_mat).reshape((self.n_classes, self.n_classes))
