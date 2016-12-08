#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Módulo para llevar a cabo las visualizaciones en Learninspy."""

__author__ = 'leferrad'

from learninspy.core.activations import fun_activation, fun_activation_d
from learninspy.core.autoencoder import StackedAutoencoder

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_matrix(matrix, ax=None, values=True, show=True):
    """
    Ploteo de un arreglo 2-D.

    :param matrix: numpy.array o list, arreglo a graficar.
    :param ax: matplotlib.axes.Axes donde se debe plotear. Si es *None*, se crea una instancia de ello.
    :param values: bool, para indicar si se desea imprimir en cada celda el valor correspondiente.
    :param show: bool, para indicar si se debe imprimir inmediatamente en pantalla mediante **matplotlib.pyplot.show()**.
    """

    if type(matrix) is list:
        matrix = np.array(matrix)

    m, n = matrix.shape
    total_row = map(lambda row: sum(row), matrix)

    if ax is None:
        # Configuro el ploteo
        fig, ax = plt.subplots()
        ax.set_title('Matrix', color='b')
        plt.setp(ax, xticks=range(n), yticks=range(m), xlabel='X', ylabel='Y')

    # Matriz de confusion final normalizada (para pintar las celdas)
    normalized = map(lambda (row, tot): [r / (tot * 1.0) for r in row], zip(matrix, total_row))

    res = ax.imshow(normalized, cmap=plt.get_cmap('YlGn'), interpolation='nearest', aspect='auto')  # Dibujo grilla con colores

    if values is True:
        # Agrego numeros en celdas
        for x in xrange(m):
            for y in xrange(n):
                ax.annotate(str(matrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    #fig.colorbar(res, fraction=0.05)
    #plt.tight_layout()
    if show is True:
        plt.show()
    return


def plot_confusion_matrix(matrix, show=True):
    """
    Ploteo de una matrix de confusión, realizada mediante la función
    :func:`~learninspy.utils.evaluation.ClassificationMetrics.confusion_matrix`.

    :param matrix: numpy.array
    """
    m, n = matrix.shape
    fig, ax = plt.subplots()
    ax.set_title('Confusion Matrix', color='g')
    plt.setp(ax, xticks=range(n), yticks=range(m), xlabel='Actual', ylabel='Predicted')
    plot_matrix(matrix, ax, values=True, show=show)


def plot_autoencoders(network, show=True):
    """
    Ploteo de la representación latente un StackedAutoencoder dado.

    .. note:: Experimental

    :param network: red neuronal, del tipo :class:`.StackedAutoencoder`.
    :param show: bool, para indicar si se debe imprimir inmediatamente en pantalla mediante **matplotlib.pyplot.show()**
    """

    n_layers = len(network.list_layers)
    # Configuro el ploteo
    gs = gridspec.GridSpec(n_layers, 2)  # N Autoencoders, 2 graficos (W, b)
    for l in xrange(len(network.list_layers) - 1):
        ae = network.list_layers[l]

        # Preparo plot de los pesos W del AutoEncoder
        ax_w = plt.subplot(gs[l, 0])
        ax_w.set_title('AE'+str(l+1)+'_W', color='r')
        plt.setp(ax_w, xlabel='j', ylabel='i')
        ax_w.get_xaxis().set_visible(False)
        ax_w.get_yaxis().set_visible(False)

        # Preparo plot del bias b del AutoEncoder
        ax_b = plt.subplot(gs[l, 1])
        ax_b.set_title('AE'+str(l+1)+'_b', color='r')
        plt.setp(ax_b, ylabel='i')
        ax_b.get_xaxis().set_visible(False)
        ax_b.get_yaxis().set_visible(False)

        # Ploteo
        plot_matrix(ae.encoder_layer().weights.matrix, ax_w, values=False, show=False)
        plot_matrix(ae.encoder_layer().bias.matrix.T, ax_b, values=False, show=False)

    if show is True:
        plt.show()


def plot_neurons(network, show=True):
    """
    Ploteo de la representación latente de una Red Neuronal.
    .. note:: Experimental

    :param network: red neuronal del tipo :class:`.NeuralNetwork`.
    :param show: bool, para indicar si se debe imprimir inmediatamente en pantalla mediante **matplotlib.pyplot.show()**
    """
    if type(network) is StackedAutoencoder:
        plot_autoencoders(network, show=show)
    else:
        n_layers = len(network.list_layers)
        # Configuro el ploteo
        gs = gridspec.GridSpec(n_layers, 2)  # N capas, 2 graficos (W, b)
        for l in xrange(len(network.list_layers)):
            layer = network.list_layers[l]

            # Preparo plot de W
            ax_w = plt.subplot(gs[l, 0])
            ax_w.set_title('W'+str(l+1), color='r')
            plt.setp(ax_w, xlabel='j', ylabel='i')
            ax_w.get_xaxis().set_visible(False)
            ax_w.get_yaxis().set_visible(False)

            # Preparo plot de b
            ax_b = plt.subplot(gs[l, 1])
            ax_b.set_title('b'+str(l+1), color='r')
            plt.setp(ax_b, ylabel='i')
            ax_b.get_xaxis().set_visible(False)
            ax_b.get_yaxis().set_visible(False)

            # Ploteo
            plot_matrix(layer.weights.matrix, ax_w, values=False, show=False)
            plot_matrix(layer.bias.matrix.T, ax_b, values=False, show=False)

        if show is True:
            plt.show()


def plot_activations(params, show=True):
    """
    Ploteo de las activaciones establecidas para una red neuronal. Se representan como señales 1-D, en un dominio dado.

    .. note:: Experimental

    :param params: parámetros del tipo :class:`.NetworkParameters`.
    :param show: bool, para indicar si se debe imprimir inmediatamente en pantalla mediante **matplotlib.pyplot.show()**
    """
    # Si la activacion es la misma para todas las capas, la ploteo una sola vez
    if all(act == params.activation[0] for act in params.activation):
        n_act = 1
    else:
        n_act = len(params.activation)

    # Configuro el ploteo
    gs = gridspec.GridSpec(n_act, 2) # N activaciones, 2 graficos (act, d_act)
    x_axis = [i / 10.0 for i in range(-50, 50)]  # Rango de -5 a 5 con 0.1 de step

    for n in xrange(n_act):
        # Grafico de act y d_act para activacion n
        ax_act = plt.subplot(gs[n, 0])
        ax_act.set_title(params.activation[n], color='r')
        ax_d_act = plt.subplot(gs[n, 1])
        ax_d_act.set_title('d_'+params.activation[n], color='r')
        # Calculo activacion y su derivada sobre valores de x
        act = [fun_activation[params.activation[n]](x) for x in x_axis]
        d_act = [fun_activation_d[params.activation[n]](x) for x in x_axis]
        # Ploteo
        ax_act.plot(x_axis, act)
        ax_d_act.plot(x_axis, d_act)
    if show is True:
        plt.show()


def plot_fitting(network, show=True):
    """
    Ploteo del ajuste obtenido en el entrenamiento de un modelo, utilizando la información
    almacenada en dicha instancia.

    :param network: red neuronal del tipo :class:`.NeuralNetwork`.
    :param show: bool, para indicar si se debe imprimir inmediatamente en pantalla mediante **matplotlib.pyplot.show()**
    """
    x = network.epochs
    y_train = network.hits_train
    y_valid = network.hits_valid
    ax = plt.subplot()
    ax.set_title("Ajuste de la red durante entrenamiento", color='b')
    plt.setp(ax, xlabel='Epochs', ylabel='Hits')
    plt.xlim([x[0] - 1, x[-1] + 1])
    plt.ylim([0, 1])
    ax.plot(x, y_train, 'bs-', label='Train')
    ax.plot(x, y_valid, 'g^-', label='Valid')
    ax.legend(loc='upper left', shadow=True, fancybox=True)
    if show is True:
        plt.show()