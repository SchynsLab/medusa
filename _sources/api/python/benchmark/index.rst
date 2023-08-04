:py:mod:`medusa.benchmark`
==========================

.. py:module:: medusa.benchmark


Module Contents
---------------

.. py:class:: FancyTimer

   Fancy timer to time Python functions.

   .. py:method:: iter(params, pbar=True)
      :staticmethod:

      Iterates over a dictionary of parameters.

      :param params: Dictionary of parameters to iterate over
      :type params: dict
      :param pbar: Whether to show a progress bar
      :type pbar: bool

      :Yields: *Dictionary of flattened parameters (from all possible combinations)*


   .. py:method:: time(f, args, n_warmup=2, repeats=10, params=None)

      Time a function for a number of repetitions.

      :param f: Function to time
      :type f: func
      :param args: Arguments to pass to ``f``
      :type args: iterable
      :param n_warmup: Number of warmup calls to ``f``
      :type n_warmup: int
      :param repeats: Number of times the function ``f`` is called
      :type repeats: int
      :param params: Extra parameters to add to output dictionary
      :type params: dict

      .. rubric:: Examples

      >>> timer = FancyTimer()
      >>> timer.time(lambda x: x ** 2, [5])
      >>> timer


   .. py:method:: to_df()

      Converts the timer results to a pandas DataFrame.



