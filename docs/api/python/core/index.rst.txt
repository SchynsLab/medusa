:py:mod:`medusa.core`
=====================

.. py:module:: medusa.core

.. autoapi-nested-parse::

   The ``core`` module contains two submodules, ``fourD`` and ``threeD`` that contain
   classes to represent and manipulate 4D and 3D data, respectively. Data from different topologies
   are represented by different classes, which all inherit from a common base class (e.g., ``Base4D``
   for 4D data). For example, data reconstructed using the Mediapipe model is stored in an object of the
   ``Mediapipe4D`` class.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   fourD/index.rst
   threeD/index.rst


Package Contents
----------------

.. py:data:: MODEL2CLS
   

   Allows to map string-based names of classes to the actual class.

.. py:data:: FLAME_MODELS
   :annotation: = ['deca-coarse', 'deca-dense', 'emoca-coarse', 'emoca-dense', 'spectre-coarse', 'spectre-dense']

   

