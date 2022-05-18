:py:mod:`medusa.data`
=====================

.. py:module:: medusa.data


Package Contents
----------------

.. py:function:: get_example_frame()

   Loads an example frame from the example video.

   :param as_path: Returns the path as a ``pathlib.Path`` object
   :type as_path: bool

   :returns: **img** -- A 3D numpy array of shape frame width x height x 3 (RGB)
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> img = get_example_frame()


.. py:function:: get_example_video(as_path=True)

   Retrieves the path to an example video file.

   :param as_path: Returns the path as a ``pathlib.Path`` object
   :type as_path: bool

   :returns: **path** -- A string or Path object pointing towards the example
             video file
   :rtype: str, pathlib.Path

   .. rubric:: Examples

   >>> path = get_example_video(as_path=True)
   >>> path.is_file()
   True


.. py:function:: get_example_h5(load=False, model='mediapipe', as_path=True)

   Retrieves an example hdf5 file with reconstructed 4D
   data from the example video.

   :param load: Whether to return the hdf5 file loaded in memory (``True``)
                or to just return the path to the file
   :type load: bool
   :param model: Model used to reconstruct the data; either 'mediapipe' or
                 'emoca'
   :type model: str
   :param as_path: Whether to return the path as a ``pathlib.Path`` object (``True``)
                   or just a string (``False``); ignored when ``load`` is ``True``
   :type as_path: bool

   :returns: When ``load`` is ``True``, returns either a ``MediapipeData``
             or a ``FlameData`` object, otherwise a string or ``pathlib.Path``
             object to the file
   :rtype: MediapipeData, FlameData, str, Path

   .. rubric:: Examples

   >>> path = get_example_h5(load=False, as_path=True)
   >>> path.is_file()
   True

   # Get hdf5 file already loaded in memory
   >>> data = get_example_h5(load=True, model='mediapipe')
   >>> data.recon_model_name
   'mediapipe'
   >>> data.v.shape  # check out reconstructed vertices
   (232, 468, 3)


