:py:mod:`medusa.epochs`
=======================

.. py:module:: medusa.epochs


Module Contents
---------------

.. py:class:: EpochsArray(v, params, frame_t, recon_model_name, events=None)

   Custom EpochsArray, with some extra functionality to interact with
   medusa.

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

   .. py:method:: save(self, path, compression_level=9)

      Saves (meta)data to disk as an HDF5 file.

      :param path: Path to save the data to
      :type path: str
      :param compression_level: Level of compression (higher = more compression, but slower; max = 9)
      :type compression_level: int


   .. py:method:: to_mne(self, frame_t, include_global_motion=True)

      Initalize a MNE EpochsArray.

      :param include_global_motion: Whether to add global motion ('mat') to the data as if it were a separate
                                    set of channels
      :type include_global_motion: bool

      :rtype: An instance of the EpochsArray class



