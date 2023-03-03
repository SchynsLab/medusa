:py:mod:`medusa.epoch`
======================

.. py:module:: medusa.epoch


Module Contents
---------------

.. py:class:: EpochsArray(v_epochs, params_epochs, frame_t)

   .. py:method:: baseline_normalize()


   .. py:method:: to_mne()


   .. py:method:: from_4D(data, events, start=-0.5, end=5.0, period=0.01, T=50, anchor='onset')
      :classmethod:


   .. py:method:: to_4D(agg='mean', device='cuda')
