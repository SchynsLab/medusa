:py:mod:`gmfx.preproc.filter`
=============================

.. py:module:: gmfx.preproc.filter


Module Contents
---------------

.. py:data:: logger
   

   

.. py:function:: filter(data, low_pass, high_pass)

   Applies a bandpass filter the vertex coordinate time series.
   Implementation based on https://stackoverflow.com/questions/
   12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

   :param data: Either a path (str, pathlib.Path) to a `gmfx` hdf5 data file
                or a gmfx.io.Data object (i.e., data loaded from the hdf5 file)
   :type data: str, Data
   :param low_pass: Low-pass cutoff (in Hz)
   :type low_pass: float
   :param high_pass: High-pass cutoff (in Hz)
   :type high_pass: float
   :param video: Path to video to render reconstruction on top of
                 (optional)
   :type video: str


