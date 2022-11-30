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


.. py:function:: get_template_flame(dense=False)

   Returns the template (vertices and triangles) of the canonical Flame model, in
   either its dense or coarse version. Note that this does exactly the same as the
   ``get_flame_template()`` function from the ``flame.data`` module.

   :param dense: Whether to load in the dense version of the template (``True``) or the coarse
                 version (``False``)
   :type dense: bool

   :raises ValueError: If the 'flame' package is not installed and/or the Flame file could not be found

   :returns: **template** -- Dictionary with vertices ("v") and faces ("f")
   :rtype: dict

   .. rubric:: Examples

   Get the vertices and faces (triangles) of the standard Flame topology (template) in
   either the coarse version (``dense=False``) or dense version (``dense=True``)

   >>> template = get_template_flame(dense=False)  # doctest: +SKIP
   >>> template['v'].shape  # doctest: +SKIP
   (5023, 3)
   >>> template['f'].shape  # doctest: +SKIP
   (9976, 3)
   >>> template = get_template_flame(dense=True)  # doctest: +SKIP
   >>> template['v'].shape  # doctest: +SKIP
   (59315, 3)
   >>> template['f'].shape  # doctest: +SKIP
   (117380, 3)


