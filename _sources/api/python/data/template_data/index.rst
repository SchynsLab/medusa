:py:mod:`medusa.data.template_data`
===================================

.. py:module:: medusa.data.template_data

.. autoapi-nested-parse::

   This module contains functions to load in "template data", i.e., the
   topological templates used by the different models.



Module Contents
---------------

.. py:function:: get_template_mediapipe(device=None)

   Returns the template (vertices and triangles) of the canonical Mediapipe
   model.

   :returns: **template** -- Dictionary with vertices ("v") and faces ("f")
   :rtype: dict

   .. rubric:: Examples

   Get the vertices and faces (triangles) of the standard Mediapipe topology (template):

   >>> template = get_template_mediapipe()
   >>> template['v'].shape
   (468, 3)
   >>> template['f'].shape
   (898, 3)


.. py:function:: get_template_flame(topo='coarse', keys=None, device=None)

   Returns the template (vertices and triangles) of the canonical Flame
   model, in either its dense or coarse version. Note that this does exactly
   the same as the ``get_flame_template()`` function from the ``flame.data``
   module.

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


.. py:function:: get_external_data_config(key=None)

   Loads the FLAME config file (i.e., the yaml with paths to the FLAME-
   based models & data.

   :param key: If ``None`` (default), the config is returned as a dictionary;
               if ``str``, then the value associated with the key is returned
   :type key: str

   :returns: The config file as a dictionary if ``key=None``, else a string
             with the value associated with the key
   :rtype: dict, str

   .. rubric:: Examples

   Load in the entire config file as a dictionary

   >>> cfg = get_external_data_config()
   >>> isinstance(cfg, dict)
   True

   Get the path of the FLAME model:

   >>> flame_path = get_external_data_config(key='flame_path')


.. py:function:: get_rigid_vertices(topo, device=DEVICE)

   Gets the default 'rigid' vertices (i.e., vertices that can only move
   rigidly) for a given topology ('mediapipe', 'flame-coarse', 'flame-dense').

   :param topo: Topology name ('mediapipe', 'flame-coarse', or 'flame-dense')
   :type topo: str
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   :returns: **v_idx** -- A long tensor with indices of rigid vertices
   :rtype: torch.tensor


.. py:function:: get_vertex_template(topo, device=DEVICE)

   Gets the default vertices (or 'template') for a given topology
   ('mediapipe', 'flame-coarse', 'flame-dense').

   :param topo: Topology name ('mediapipe', 'flame-coarse', or 'flame-dense')
   :type topo: str
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   :returns: **target** -- A float tensor with the default (template) vertices
   :rtype: torch.tensor


.. py:function:: get_tris(topo, device=DEVICE)

   Gets the triangles for a given topology ('mediapipe', 'flame-coarse',
   'flame-dense').

   :param topo: Topology name ('mediapipe', 'flame-coarse', or 'flame-dense')
   :type topo: str
   :param device: Either 'cuda' (GPU) or 'cpu'
   :type device: str

   :returns: **tris** -- A long tensor with the triangles
   :rtype: torch.tensor


