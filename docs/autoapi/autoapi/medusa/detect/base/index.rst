:py:mod:`medusa.detect.base`
============================

.. py:module:: medusa.detect.base


Module Contents
---------------

.. py:class:: BaseDetector

   .. py:method:: detect_faces_video(vid, batch_size=32)

      Utility function to get all detections in a video.

      :param vid: Path to video (or, optionally, a ``VideoLoader`` object)
      :type vid: str, Path

      :returns: **results** -- A BatchResults object with all detection information
      :rtype: BatchResults
