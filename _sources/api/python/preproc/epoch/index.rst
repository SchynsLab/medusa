:py:mod:`medusa.preproc.epoch`
==============================

.. py:module:: medusa.preproc.epoch


Module Contents
---------------

.. py:function:: epoch(data, start=-0.5, end=3.0, period=0.01, align_peaks=False, max_shift=0.5, baseline_correct=False, baseline_window=(None, None), baseline_mode='mean')

   Creates epochs of the data.

   :param data: Either a path (``str`` or ``pathlib.Path``) to a ``medusa`` hdf5
                data file or a ``Data`` object (like ``FlameData`` or ``MediapipeData``)
   :type data: str, Data
   :param start: Start of the epoch (in seconds) relative to stimulus onset
   :type start: float
   :param end: End of the epoch (in seconds) relative to stimulus onset
   :type end: float
   :param align_peaks: Whether to align peaks across epochs (not implemented yet)
   :type align_peaks: bool
   :param max_shift: Maximum allowed shift (in seconds) for peak alignment; if `peak_align`
                     is False, then this argument is ignored
   :type max_shift: float
   :param baseline_correct: Whether to apply baseline correction
   :type baseline_correct: bool
   :param baseline_window: Tuple with two values, indicating baseline start and end (in seconds),
                           respectively; if the first value is None, then the start is the beginning
                           of the epoch; if the second value is None, then the end is at stimulus onset
                           (i.e., 0)
   :type baseline_window: tuple[float]
   :param baseline_mode: How to perform baseline correction (options: 'mean', 'ratio')
   :type baseline_mode: str


