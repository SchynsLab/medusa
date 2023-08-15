:py:mod:`medusa.epoch`
======================

.. py:module:: medusa.epoch


Module Contents
---------------

.. py:class:: EpochsArray(v_epochs, params_epochs, frame_t)

   Experimental class for epoching and storing epoched data.

   :param v_epochs: Array with epoched vertices
   :type v_epochs: np.ndarray
   :param params_epochs: Array with epoched global movement parameters
   :type params_epochs: np.ndarray
   :param frame_t: Array with frame times
   :type frame_t: np.ndarray

   .. py:method:: baseline_normalize()


   .. py:method:: to_mne()


   .. py:method:: from_4D(data, events, start=-0.5, end=5.0, period=0.01, T=50, anchor='onset', anchor_units='frames')
      :classmethod:

      A classmethod to create an ``EpochsArray`` object from a single `Data4D` object
      (with multiple events) or a list of `Data4D` objects (with one event each).

      :param data: A single `Data4D` object or a list of `Data4D` objects
      :type data: Data4D, list[Data4D]
      :param events: Path to a TSV file with events or a pandas DataFrame with events; should have
                     one of the columns ['onset', 'offset'] and optionally ['duration'], which
                     depends on your choice of `anchor` (see below)
      :type events: str, Path, pd.DataFrame
      :param start: Start time of the epoch relative to the anchor (in seconds)
      :type start: float
      :param end: End time of the epoch relative to the anchor (in seconds)
      :type end: float
      :param period: Sampling period of the epoch (in seconds)
      :type period: float
      :param T: Number of time points in the epoch (only relevant when `anchor` is not
                'duration' or 'span')
      :type T: int
      :param anchor: Anchor point of the epoch; one of ['onset', 'offset', 'duration', 'span']
                     (where 'span' represents the entire period between 'onset' and 'offset')
      :type anchor: str
      :param anchor_units: Units of the anchor point; one of ['frames', 'seconds']
      :type anchor_units: str


   .. py:method:: to_4D(agg='mean', cam_mat=None, device=DEVICE, **kwargs)

      Converts the `EpochsArray` object to a `Data4D` object by aggregating
      (e.g., averaging) the epochs.

      :param agg: Aggregation method; one of ['mean', 'median']
      :type agg: str
      :param cam_mat: Camera matrix (e.g., take one from the inputs `Data4D` objects)
      :type cam_mat: np.ndarray
      :param device: Either 'cpu' or 'cuda' (GPU)
      :type device: str
      :param kwargs: Additional keyword arguments to pass to the `Data4D` constructor
      :type kwargs: dict

      :returns: **data** -- A `Data4D` object with the aggregated epochs
      :rtype: Data4D



