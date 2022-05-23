:py:mod:`medusa.preproc.align`
==============================

.. py:module:: medusa.preproc.align


Module Contents
---------------

.. py:function:: align(data, algorithm='icp', additive_alignment=False, ignore_existing=False, reference_index=0)

   Aligment of 3D meshes over time.

   :param data: Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
                data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
   :type data: str, Data
   :param algorithm: Either 'icp' or 'umeyama'
   :type algorithm: str
   :param qc: Whether to visualize a quality control plot
   :type qc: bool
   :param additive_alignment: Whether to estimate an additional set of alignment parameters on
                              top of the existing ones (if present; ignored otherwise)
   :type additive_alignment: bool
   :param ignore_existing: Whether to ignore the existing alignment parameters
   :type ignore_existing: bool
   :param reference_index: Index of the mesh used as the reference mesh; for reconstructions that already
                           include the local-to-world matrix, the reference mesh is only used to fix the
                           camera to; for other reconstructions, the reference mesh is used as the target
                           to align all other meshes to
   :type reference_index: int

   :returns: **data** -- An object with a class inherited from ``medusa.core.BaseData``
   :rtype: medusa.core.*Data


