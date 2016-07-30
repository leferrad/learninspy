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
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.neurons
------------------------------

.. automodule:: learninspy.core.neurons
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.optimization
-----------------------------------
Este módulo se realizó en base al excelente package de optimización `climin <https://github.com/BRML/climin>`_ , de donde se adaptaron algunos algoritmos de optimización para su uso en redes neuronales. 

.. note:: Proximamente se migrará a un package *optimization*, separando por scripts los algoritmos de optimización.


.. automodule:: learninspy.core.optimization
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.search
-----------------------------

.. automodule:: learninspy.core.search
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.stops
----------------------------

.. automodule:: learninspy.core.stops
    :members:
    :undoc-members:
    :show-inheritance:

