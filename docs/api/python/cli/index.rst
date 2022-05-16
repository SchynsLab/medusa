:py:mod:`gmfx.cli`
==================

.. py:module:: gmfx.cli


Module Contents
---------------

.. py:function:: videorecon_cmd(video_path, events_path, recon_model_name, cfg, device, out_dir, render_recon, render_on_video, render_crop, n_frames)


.. py:function:: align_cmd(data, algorithm, qc)


.. py:function:: resample_cmd(data, sampling_freq, kind)


.. py:function:: filter_cmd(data, low_pass, high_pass)


.. py:function:: epoch_cmd(data, start, end, period)


.. py:function:: videorender_cmd(h5_path, video, n_frames, no_smooth, wireframe, alpha, scaling, fmt)

   Renders the reconstructed vertices as a video.


