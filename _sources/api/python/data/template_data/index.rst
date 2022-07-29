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

   .. rubric:: Examples

   Get the vertices and faces (triangles) of the standard Mediapipe topology (template):

   >>> template = get_template_mediapipe()
   >>> template['v'].shape
   (468, 3)
   >>> template['f'].shape
   (898, 3)


.. py:function:: get_template_flame()

   Returns the template (vertices and triangles) of the canonical Flame model.

   :raises ValueError: If the 'flame' package is not installed and/or the Flame file could not be found

   :returns: **template** -- Dictionary with vertices ("v") and faces ("f")
   :rtype: dict

   .. rubric:: Examples

   Get the vertices and faces (triangles) of the standard Flame topology (template):

   >>> template = get_template_mediapipe()  # doctest: +SKIP
   >>> template['v'].shape  # doctest: +SKIP
   (468, 3)
   >>> template['f'].shape  # doctest: +SKIP
   (898, 3)


