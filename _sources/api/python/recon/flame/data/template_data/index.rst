:py:mod:`medusa.recon.flame.data.template_data`
===============================================

.. py:module:: medusa.recon.flame.data.template_data


Module Contents
---------------

.. py:function:: get_template_flame(dense=False)

   Returns the template (vertices and triangles) of the canonical Flame model, in
   either its dense or coarse version.

   :param dense: Whether to load in the dense version of the template (``True``) or the coarse
                 version (``False``)
   :type dense: bool

   :returns: **template** -- Dictionary with vertices ("v") and faces ("f")
   :rtype: dict

   .. rubric:: Examples

   Get the vertices and faces (triangles) of the standard Flame topology (template) in
   either the coarse version (``dense=False``) or dense version (``dense=True``)

   >>> template = get_template_flame(dense=False)
   >>> template['v'].shape
   (5023, 3)
   >>> template['f'].shape
   (9976, 3)
   >>> template = get_template_flame(dense=True)
   >>> template['v'].shape
   (59315, 3)
   >>> template['f'].shape
   (117380, 3)


