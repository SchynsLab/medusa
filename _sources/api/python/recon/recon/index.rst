:py:mod:`medusa.recon.recon`
============================

.. py:module:: medusa.recon.recon

.. autoapi-nested-parse::

   Module with canonical ``videorecon`` function that takes in a video and
   returns a ``Data4D`` object.



Module Contents
---------------

.. py:function:: videorecon(video_path, recon_model='mediapipe', device=DEVICE, n_frames=None, batch_size=32, loglevel='INFO', **kwargs)

   Reconstruction of all frames of a video.

   :param video_path: Path to video file to reconstruct, or an already initialized VideoLoader
   :type video_path: str, Path, VideoLoader
   :param recon_model: Name of reconstruction model, options are: 'deca-coarse', 'deca-dense',
                       'emoca-coarse', 'emoca-dense', and 'mediapipe'
   :type recon_model: str
   :param device: Either "cuda" (for GPU) or "cpu"
   :type device: str
   :param n_frames: If not ``None`` (default), only reconstruct and render the first ``n_frames``
                    frames of the video; nice for debugging
   :type n_frames: int
   :param batch_size: Batch size (i.e., number of frames) processed by the reconstruction model
                      in each iteration; decrease this number when you get out of memory errors
   :type batch_size: int
   :param loglevel: Logging level, options are (in order of verbosity): 'DEBUG', 'INFO', 'WARNING',
                    'ERROR', and 'CRITICAL'
   :type loglevel: str
   :param \*\*kwargs: Additional keyword arguments passed to the reconstruction model initialization

   :returns: **data** -- An Data4D object with all reconstruction (meta)data
   :rtype: Data4D

   .. rubric:: Examples

   Reconstruct a video using Mediapipe:

   >>> from medusa.data import get_example_video
   >>> vid = get_example_video()
   >>> data = videorecon(vid, recon_model='mediapipe')


