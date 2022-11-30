:py:mod:`medusa.recon.recon`
============================

.. py:module:: medusa.recon.recon


Module Contents
---------------

.. py:function:: videorecon(video_path, recon_model='mediapipe', device=DEVICE, n_frames=None, batch_size=32, loglevel='INFO')

   Reconstruction of all frames of a video.

   :param video_path: Path to video file to reconstruct
   :type video_path: str, Path
   :param events_path: Path to events file (a TSV or CSV file) containing
                       info about experimental events; must have at least the
                       columns 'onset' (in seconds) and 'trial_type'; optional
                       columns are 'duration' and 'modulation'
   :type events_path: str, Path
   :param recon_model: Name of reconstruction model, options are: 'emoca', 'mediapipe',
   :type recon_model: str
   :param device: Either "cuda" (for GPU) or "cpu" (ignored when using mediapipe)
   :type device: str
   :param n_frames: If not `None` (default), only reconstruct and render the first `n_frames`
                    frames of the video; nice for debugging
   :type n_frames: int
   :param batch_size: Batch size (i.e., number of frames) processed by the reconstruction model
                      in each iteration; decrease this number when you get out of memory errors
   :type batch_size: int
   :param loglevel: Logging level, options are (in order of verbosity): 'DEBUG', 'INFO', 'WARNING',
                    'ERROR', and 'CRITICAL'
   :type loglevel: str

   :returns: **data** -- An object with a class inherited from ``medusa.core.BaseData``
   :rtype: medusa.core.*Data

   .. rubric:: Examples

   Reconstruct a video using Mediapipe:

   >>> from medusa.data import get_example_video
   >>> vid = get_example_video()
   >>> data = videorecon(vid, recon_model='mediapipe')


