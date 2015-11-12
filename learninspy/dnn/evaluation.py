__author__ = 'leferrad'

# Dependencias externas
import numpy as np

class ClassificationMetrics(object):
    # Ver http://machine-learning.tumblr.com/post/1209400132/mathematical-definitions-for-precisionrecall-for
    # Ver http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf
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

    def precision(self, label=None, macro=True):
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

    def recall(self, label=None, macro=True):
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

    def f_measure(self, beta=1, label=None, macro=True):
        ppv = self.precision(label, macro)
        tpr = self.recall(label, macro)
        f_score = (1 + beta*beta)*(ppv * tpr) / (beta*beta*ppv + tpr)
        return f_score

    def confusion_matrix(self):
        conf_mat = []  # Matriz de confusion final
        for r in xrange(self.n_classes):
            pre_act = filter(lambda (p, a): a == r, self.predicted_actual)
            for c in xrange(self.n_classes):
                conf_mat.append(sum(map(lambda (p, a): p == c, pre_act)))
        return np.array(conf_mat).reshape((self.n_classes, self.n_classes))

class RegressionMetrics(object):
    def __init__(self, predicted_actual):
        self.predicted_actual = predicted_actual
        self.n_elem = len(predicted_actual)
        self.error = map(lambda (p, a): a - p, self.predicted_actual)

    def mse(self):
        return np.sum(np.square(self.error)) / float(self.n_elem)

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return np.sum(np.abs(self.error))

    def r2(self):
        # Ver https://en.wikipedia.org/wiki/Coefficient_of_determination
        mean_actual = np.mean(map(lambda (p, a): a, self.predicted_actual))
        ssres = np.sum(np.square(self.error))
        sstot = np.sum(np.square(map(lambda (p, a): a - mean_actual, self.predicted_actual)))
        return 1 - float(ssres / sstot)

    def explained_variance(self):
        var_error = np.var(self.error)
        var_actual = np.var(map(lambda (p, a): a, self.predicted_actual))
        return 1 - float(var_error / var_actual)
