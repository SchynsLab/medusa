:py:mod:`gmfx.preproc.align`
============================

.. py:module:: gmfx.preproc.align


Module Contents
---------------

.. py:data:: logger
   

   

.. py:function:: align(data, algorithm, qc=False, additive_alignment=False, ignore_existing=False)

   Aligment of 3D meshes over time.

   :param data: Either a path (str, pathlib.Path) to a `gmfx` hdf5 data file
                or a gmfx.io.Data object (i.e., data loaded from the hdf5 file)
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


