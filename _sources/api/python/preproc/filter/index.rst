:py:mod:`medusa.preproc.filter`
===============================

.. py:module:: medusa.preproc.filter


Module Contents
---------------

.. py:function:: filter(data, low_pass, high_pass)

   Applies a bandpass filter the vertex coordinate time series.
   Implementation based on
   `this StackOverflow post` <https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter>`_.

   :param data: Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
                data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
   :type data: str, Data
   :param low_pass: Low-pass cutoff (in Hz)
   :type low_pass: float
   :param high_pass: High-pass cutoff (in Hz)
   :type high_pass: float

   :returns: **data** -- An object with a class inherited from ``medusa.core.BaseData``
   :rtype: medusa.core.*Data

   .. rubric:: Examples

   Filter the data wit a high-pass of 0.005 Hz and a low-pass of 4 Hz:

   >>> from medusa.data import get_example_h5
   >>> data = get_example_h5(load=True, model='mediapipe')
   >>> data = filter(data, low_pass=4., high_pass=0.005)


