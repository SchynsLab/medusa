:py:mod:`medusa.io`
===================

.. py:module:: medusa.io

.. autoapi-nested-parse::

   Module with functionality (mostly) for working with video data.
   The `VideoData` class allows for easy looping over frames of a video file,
   which is used in the reconstruction process (e.g., in the ``videorecon`` function).



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

   .. py:method:: loop(self, scaling=None, return_index=True, verbose=True)

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



.. py:data:: EpochsArrayBase
   

   

.. py:class:: EpochsArray(*args, **kwargs)

   Bases: :py:obj:`mne.epochs.EpochsArray`

   Custom EpochsArray, with some extra functionality to interact with
   medusa.

   :param args: Positional parameters to be passed to initialization of the
                MNE EPochsArray (the base class)
   :type args: list
   :param kwargs: Keyword parameters to be passed to initialization of the
                  MNE EPochsArray (the base class)
   :type kwargs: list

   .. py:method:: from_medusa(cls, v, sf, events=None, frame_t=None, tmin=-0.5, includes_motion=False)
      :classmethod:

      Classmethod to initalize an EpochsArray from medusa data.

      :param v: A 4D numpy array of shape N (events/trails) x T (time points)
                x nV (number of vertices) x 3 (X, Y, Z)
      :type v: np.ndarray
      :param sf: Sampling frequency of the data (`v`)
      :type sf: float
      :param events: events : pd.DataFrame
                     A BIDS-style DataFrame with event (trial) information,
                     with at least the columns 'onset' and 'trial_type'
      :type events: pd.DataFrame
      :param frame_t: A 1D numpy array with the onset of each frame from
                      the video that was reconstructed
      :type frame_t: np.ndarray
      :param tmin: Start (in seconds) of each epoch relative to stimulus onset
      :type tmin: float
      :param includes_motion: Whether the data (`v`) also includes the epoched motion parameters;
                              if so, it is assumed that the last 12 values in the third dimension
                              of `v` represents the motion parameters
      :type includes_motion: bool

      :rtype: An instance of the EpochsArray class


   .. py:method:: events_to_mne(events, frame_t)
      :staticmethod:

      Converts events DataFrame to (N x 3) array that
      MNE expects.

      :param events: A BIDS-style DataFrame with event (trial) information,
                     with at least the columns 'onset' and 'trial_type'
      :type events: pd.DataFrame
      :param frame_t: A 1D numpy array with the onset of each frame from
                      the video that was reconstructed; necessary for
                      converting event onsets in seconds to event onsets
                      in samples (TODO: use sf for this?)
      :type frame_t: np.ndarray

      :returns: * **events_** (*np.ndarray*) -- An N (number of trials) x 3 array, with the first column
                  indicating the sample *number* (not time) and the third
                  column indicating the sample condition (see the returned
                  `event_id` dictionary for the mapping between condition number
                  and string representation)
                * **event_id** (*dict*) -- A dictionary with condition strings as keys and condition numbers
                  as values; the values correspond to the third column of `events_`



