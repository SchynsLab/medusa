:py:mod:`medusa.preproc.recon`
==============================

.. py:module:: medusa.preproc.recon


Module Contents
---------------

.. py:data:: logger
   

   

.. py:function:: videorecon(video_path, events_path=None, recon_model_name='mediapipe', cfg=None, device='cuda', out_dir=None, render_recon=True, render_on_video=False, render_crop=False, n_frames=None)

   Reconstruction of all frames of a video.

   :param video_path: Path to video file to reconstruct
   :type video_path: str, Path
   :param events_path: Path to events file (a TSV or CSV file) containing
                       info about experimental events; must have at least the
                       columns 'onset' (in seconds) and 'trial_type'; optional
                       columns are 'duration' and 'modulation'
   :type events_path: str, Path
   :param recon_model_name: Name of reconstruction model, options are: 'emoca', 'mediapipe',
                            and 'FAN-3D'
   :type recon_model_name: str
   :param cfg: Path to config file for EMOCA reconstruction; ignored if not using emoca
   :type cfg: str
   :param device: Either "cuda" (for GPU) or "cpu" (ignored when using mediapipe)
   :type device: str
   :param out_dir: Path to directory where recon data (and associated
                   files) are saved; if `None`, same directory as video is used
   :type out_dir: str, Path
   :param render_on_video: Whether to render the reconstruction on top of the video;
                           this may substantially increase rendering time!
   :type render_on_video: bool
   :param render_crop: Whether to render the cropping results (only relevant when using EMOCA,
                       ignored otherwise)
   :type render_crop: bool
   :param n_frames: If not `None` (default), only reconstruct and render the first `n_frames`
                    frames of the video; nice for debugging
   :type n_frames: int


