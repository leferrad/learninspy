learninspy.core package
=======================

Éste es el módulo principal o el núcleo del framework, y contiene clases relacionadas con la construcción de redes neuronales profundas, desde el diseño de la arquitectura hasta la optimización del desempeño en las tareas asignadas.

Submódulos
----------

learninspy.core.activations
----------------------------------

En este módulo se pueden configurar las funciones de activación que se deseen. Para ello, simplemente se codifica tanto la función como su derivada analítica (o aproximación, como en el caso de la ReLU), y luego se insertan en los diccionarios de funciones correspondientes, que se encuentran al final del script, con una key común que identifique la activación.

.. automodule:: learninspy.core.activations
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.autoencoder
----------------------------------

.. automodule:: learninspy.core.autoencoder
    :members:
    :undoc-members:
    :show-inheritance:

learninspy.core.loss
---------------------------
En este módulo se proveen dos funciones de costo populares, cuyo uso se corresponde a la tarea designada para el modelo:

* **Clasificación**: Entropía Cruzada (en inglés, *Cross Entropy o CE*), 
* **Regresión**: Error Cuadrático Medio (en inglés, *Mean Squared Error o MSE*).

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

