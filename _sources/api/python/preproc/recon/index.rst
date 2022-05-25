:py:mod:`medusa.preproc.recon`
==============================

.. py:module:: medusa.preproc.recon


Module Contents
---------------

.. py:function:: videorecon(video_path, events_path=None, recon_model_name='mediapipe', cfg=None, device='cuda', n_frames=None, loglevel='INFO')

   Reconstruction of all frames of a video.

   :param video_path: Path to video file to reconstruct
   :type video_path: str, Path
   :param events_path: Path to events file (a TSV or CSV file) containing
                       info about experimental events; must have at least the
                       columns 'onset' (in seconds) and 'trial_type'; optional
                       columns are 'duration' and 'modulation'
   :type events_path: str, Path
   :param recon_model_name: Name of reconstruction model, options are: 'emoca', 'mediapipe',
                            and 'fan'
   :type recon_model_name: str
   :param cfg: Path to config file for EMOCA reconstruction; ignored if not using emoca
   :type cfg: str
   :param device: Either "cuda" (for GPU) or "cpu" (ignored when using mediapipe)
   :type device: str
   :param n_frames: If not `None` (default), only reconstruct and render the first `n_frames`
                    frames of the video; nice for debugging
   :type n_frames: int
   :param loglevel: Logging level, options are (in order of verbosity): 'DEBUG', 'INFO', 'WARNING',
                    'ERROR', and 'CRITICAL'
   :type loglevel: str

   :returns: **data** -- An object with a class inherited from ``medusa.core.BaseData``
   :rtype: medusa.core.*Data

   .. rubric:: Examples

   Reconstruct a video using Mediapipe:

   >>> from medusa.data import get_example_video
   >>> vid = get_example_video()
   >>> data = videorecon(vid, recon_model_name='mediapipe')

   Reconstruct a video using FAN, but only the first 50 frames of the video:

   >>> data = videorecon(vid, recon_model_name='fan', n_frames=50, device='cpu')
   >>> data.v.shape
   (50, 68, 3)


