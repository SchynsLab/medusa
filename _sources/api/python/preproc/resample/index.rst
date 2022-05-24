:py:mod:`medusa.preproc.resample`
=================================

.. py:module:: medusa.preproc.resample


Module Contents
---------------

.. py:function:: resample(data, sampling_freq=None, kind='pchip')

   Resamples the data to a given sampling rate. This function can be used to
   resample the time points to a higher temporal resolution and/or a constant
   sampling period, which may not be the case for data that is acquired using a webcam.

   :param data: Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
                data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
   :type data: str, Data
   :param sampling_freq: Desired sampling frequency (in Hz); if `None` (default), the inverse
                         of the (average) sampling period will be used
   :type sampling_freq: int
   :param kind: Kind of interpolation to use, either 'pchip' (default), 'linear', 'quadratic',
                or 'cubic'
   :type kind: str

   :returns: **data** -- An object with a class inherited from ``medusa.core.BaseData``
   :rtype: medusa.core.*Data

   .. rubric:: Examples

   Resample a time series of 3D meshes to a sampling frequency of 50 Hz

   >>> from medusa.data import get_example_h5
   >>> data = get_example_h5(load=True, model='mediapipe')
   >>> data.sf  # sampling frequency
   23.98
   >>> data.v.shape[0]  # how many time points?
   232
   >>> data = resample(data, sampling_freq=50)
   >>> data.sf  # corresponds to desired sampling_freq!
   50
   >>> data.v.shape[0]  # how many time points after resampling?
   482


