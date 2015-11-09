__author__ = 'leferrad'

# Dependencias externas
#import mpld3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Librerias internas
from dnn.activations import fun_activation, fun_activation_d


def plot_matrix(matrix, xlabel='X', ylabel='Y',  name='Matrix', values=True):

        m, n = matrix.shape
        total_row = map(lambda row: sum(row), matrix)

        # Configuro el ploteo
        fig, ax = plt.subplots()
        ax.set_title(name, color='b')

        # Matriz de confusion final normalizada (para pintar las celdas)
        normalized = map(lambda (row, tot): [r / (tot * 1.0) for r in row], zip(matrix, total_row))

        res = ax.imshow(normalized, cmap=plt.get_cmap('jet'), interpolation='nearest')  # Dibujo grilla con colores

        if values is True:
            # Agrego numeros en celdas
            for x in xrange(m):
                for y in xrange(n):
                    ax.annotate(str(matrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

        #fig.colorbar(res, fraction=0.05)
        plt.setp(ax, xticks=range(n), yticks=range(m), xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        #mpld3.plugins.clear(fig)
        plt.show()
        return


def plot_confusion_matrix(matrix):
    plot_matrix(matrix, xlabel='Actual', ylabel='Predicted', name='Confusion Matrix', values=True)
    return


def plot_activations(params):
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
        ax_d_act = plt.subplot(gs[n, 1])
        # Calculo activacion y su derivada sobre valores de x
        act = [fun_activation[params.activation[n]](x) for x in x_axis]
        d_act = [fun_activation_d[params.activation[n]](x) for x in x_axis]
        # Ploteo
        ax_act.plot(x_axis, act)
        ax_d_act.plot(x_axis, d_act)
    plt.show()
    return

