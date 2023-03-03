:py:mod:`medusa.crop.base`
==========================

.. py:module:: medusa.crop.base


Module Contents
---------------

.. py:class:: BaseCropModel

   Base crop model, from which all crop models inherit.

   .. py:method:: crop_faces_video(vid, batch_size=32, save_imgs=False)

      Utility function to crop all faces in each frame of a video.

      :param vid: Path to video (or, optionally, a ``VideoLoader`` object)
      :type vid: str, Path

      :returns: **results** -- A BatchResults object with all crop information/results
      :rtype: BatchResults
