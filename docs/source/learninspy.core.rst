learninspy.core package
=======================

Éste es el módulo principal o el núcleo del framework, y contiene clases relacionadas con la construcción de redes neuronales profundas, desde el diseño de la arquitectura hasta la optimización del desempeño en las tareas asignadas.

Submódulos
----------

learninspy.core.activations
----------------------------------

.. automodule:: learninspy.core.activations

.. autofunction:: identity
.. autofunction:: tanh
.. autofunction:: sigmoid
.. autofunction:: relu
.. autofunction:: leaky_relu
.. autofunction:: softplus
.. autofunction:: lecunn_sigmoid

learninspy.core.autoencoder
----------------------------------

.. automodule:: learninspy.core.autoencoder
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.loss
---------------------------

.. automodule:: learninspy.core.loss
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.model
----------------------------

.. automodule:: learninspy.core.model

Capas neuronales
""""""""""""""""

.. autoclass:: NeuralLayer
    :show-inheritance:
.. autoclass:: ClassificationLayer
    :show-inheritance:
.. autoclass:: RegressionLayer
    :show-inheritance:

Red neuronal
""""""""""""

.. autoclass:: NeuralNetwork
    :members:
    :show-inheritance:

Parámetros
""""""""""

.. autoclass:: NetworkParameters


learninspy.core.neurons
------------------------------

.. automodule:: learninspy.core.neurons
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.optimization
-----------------------------------

.. automodule:: learninspy.core.optimization

Parámetros
""""""""""

.. autoclass:: OptimizerParameters

Algoritmos de optimización
""""""""""""""""""""""""""

.. autoclass:: Optimizer
    :show-inheritance:
.. autoclass:: GD
    :show-inheritance:
.. autoclass:: Adadelta
    :show-inheritance:

Entrenamiento distribuido
"""""""""""""""""""""""""

.. autofunction:: optimize
.. autofunction:: merge_models
.. autofunction:: mix_models

learninspy.core.search
-----------------------------

.. automodule:: learninspy.core.search
    :members:
    :undoc-members:

learninspy.core.stops
----------------------------

.. automodule:: learninspy.core.stops
    :members:
    :undoc-members:

