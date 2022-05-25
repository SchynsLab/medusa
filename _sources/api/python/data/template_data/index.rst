:py:mod:`medusa.data.template_data`
===================================

.. py:module:: medusa.data.template_data

.. autoapi-nested-parse::

   This module contains functions to load in "template data", i.e., the topological
   templates used by the different models.



Module Contents
---------------

.. py:function:: get_template_mediapipe()

   Returns the template (vertices and triangles) of the canonical Mediapipe model.

   :returns: **template** -- Dictionary with vertices ("v") and faces ("f")
   :rtype: dict


.. py:function:: get_template_flame()

   Returns the template (vertices and triangles) of the canonical Flame model.

   :returns: **template** -- Dictionary with vertices ("v") and faces ("f")
   :rtype: dict


