:py:mod:`medusa.io`
===================

.. py:module:: medusa.io

.. autoapi-nested-parse::

   Module with functionality (mostly) for working with video data.
   The `VideoData` class allows for easy looping over frames of a video file,
   which is used in the reconstruction process (e.g., in the ``videorecon`` function).



Module Contents
---------------

.. py:class:: VideoData(path, events=None, find_files=True, scaling=None, loglevel='INFO')

   " Contains (meta)data and functionality associated
   with video files (mp4 files only currently).

   :param path: Path to mp4 file
   :type path: str, Path
   :param events: Path to a TSV file with event information (optional);
                  should contain at least the columns 'onset' and 'trial_type'
   :type events: str, Path
   :param scaling: Scaling factor of video frames (e.g., 0.25 means scale to 25% of original)
   :type scaling: float
   :param loglevel: Logging level (e.g., 'INFO' or 'WARNING')
   :type loglevel: str

   .. attribute:: sf

      Sampling frequency (= frames per second, fps) of video

      :type: int

   .. attribute:: n_img

      Number of images (frames) in the video

      :type: int

   .. attribute:: img_size

      Width and height (in pixels) of the video

      :type: tuple

   .. attribute:: frame_t

      An array of length `self.n_img` with the onset of each
      frame of the video

      :type: np.ndarray

   .. py:method:: loop(return_index=True)

      Loops across frames of a video.

      :param return_index: Whether to return the frame index and the image; if ``False``,
                           only the image is returned
      :type return_index: bool

      :Yields: * **img** (*np.ndarray*) -- Numpy array (dtype: ``np.uint8``) of shape width x height x 3 (RGB)
               * **idx** (*int*) -- Optionally (when ``return_index`` is set to ``True``), returns the index of
                 the currently looped frame


   .. py:method:: stop_loop()

      Stops the loop over frames (in self.loop).


   .. py:method:: create_writer(path, idf='crop', ext='gif')

      Creates a imageio writer object, which can for example
      be used to save crop parameters on top of each frame of
      a video.


   .. py:method:: write(img)

      Adds image to writer.


   .. py:method:: get_metadata()

      Returns all (meta)data needed for initialization
      of a Data object.



.. py:function:: load_h5(path)

   Convenience function to load a hdf5 file and immediately initialize the correct
   data class.

   :param path: Path to an HDF5 file
   :type path: str

   :returns: **data** -- An object with a class derived from ``data.BaseData``
             (like ``MediapipeData``, or ``FlameData``)
   :rtype: ``data.BaseData`` subclass

   .. rubric:: Examples

   Load in HDF5 data reconstructed by Mediapipe:

   >>> from medusa.data import get_example_h5
   >>> path = get_example_h5(load=False)
   >>> data = load_h5(path)


