:py:mod:`medusa.epochs`
=======================

.. py:module:: medusa.epochs

.. autoapi-nested-parse::

   Module with functionality to store epoched data and easy interaction with
   and transfer to `MNE <https://mne.tools/stable/index.html>`_.



Module Contents
---------------

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



