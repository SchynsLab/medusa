:py:mod:`medusa.preproc.align`
==============================

.. py:module:: medusa.preproc.align


Module Contents
---------------

.. py:function:: align(data, algorithm, additive_alignment=False, ignore_existing=False)

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

   :returns: **data** -- An object with a class inherited from ``medusa.core.BaseData``
   :rtype: medusa.core.*Data


