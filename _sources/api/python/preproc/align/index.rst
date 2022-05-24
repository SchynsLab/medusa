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
   :param algorithm: Either 'icp' or 'umeyama'; ignored for Mediapipe or EMOCA reconstructions
                     (except if ``additive_alignment`` or ``ignore_existing`` is set to ``True``)
   :type algorithm: str
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

   .. rubric:: Examples

   Align sequence of 3D Mediapipe meshes using its previously estimated local-to-world
   matrices (the default alignment option):

   >>> from medusa.data import get_example_h5
   >>> data = get_example_h5(load=True, model='mediapipe')
   >>> data.space  # before alignment, data is is 'world' space
   'world'
   >>> data = align(data)
   >>> data.space  # after alignment, data is in 'local' space
   'local'

   Align sequence of 3D FAN meshes using ICP:

   >>> data = get_example_h5(load=True, model='fan')
   >>> data.mat is None  # no affine matrices yet
   True
   >>> data = align(data, algorithm='icp')
   >>> data.mat.shape  # an affine matrix for each time point!
   (232, 4, 4)

   Do an initial alignment of EMOCA meshes using the existing transform, but also
   do additional alignment (probably not a good idea):

   >>> data = get_example_h5(load=True, model='emoca')
   >>> data = align(data, algorithm='icp', additive_alignment=True)


