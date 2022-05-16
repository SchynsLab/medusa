:py:mod:`gmfx.io`
=================

.. py:module:: gmfx.io


Module Contents
---------------

.. py:data:: logger
   

   

.. py:class:: VideoData(path, events=None, find_files=True)

   " Contains (meta)data and functionality associated
   with video files (mp4 files only currently).

   :param path: Path to mp4 file
   :type path: str, Path
   :param events: Path to a TSV file with event information (optional);
                  should contain at least the columns 'onset' and 'trial_type'
   :type events: str, Path

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

   .. py:method:: loop(self, scaling=None, return_index=True)

      Loops across frames of a video.

      :param scaling: If not `None` (default), rescale image with this factor
                      (e.g., 0.25 means reduce image to 25% or original)
      :type scaling: float
      :param return_index: Whether to return the frame index and the image; if `False`,
                           only the image is returned
      :type return_index: bool

      :Yields: **img** (*np.ndarray*) -- Numpy array (dtype: `np.uint8`) of shape width x height x 3 (RGB)


   .. py:method:: stop_loop(self)

      Stops the loop over frames (in self.loop).


   .. py:method:: create_writer(self, path, idf='crop', ext='gif')

      Creates a imageio writer object, which can for example
      be used to save crop parameters on top of each frame of
      a video.


   .. py:method:: write(self, img)

      Adds image to writer.


   .. py:method:: get_metadata(self)

      Returns all (meta)data needed for initialization
      of a Data object.



