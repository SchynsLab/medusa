:py:mod:`medusa.data.example_data`
==================================

.. py:module:: medusa.data.example_data

.. autoapi-nested-parse::

   This module contains functions to load in example data, which is used for
   examples and tests. The example data is the following video:
   https://www.pexels.com/video/close-up-of-a-woman-showing-different-facial-
   expressions-3063839/ made freely available by Wolfgang Langer.

   The video was trimmed to 10 seconds and resized in order to reduce disk
   space.



Module Contents
---------------

.. py:function:: get_example_image(n_faces=None, load=True, device=DEVICE, channels_last=False, dtype=torch.float32)

   Loads an example frame from the example video.

   :param n_faces: If None, it will return the default (example) image (the first frame from
                   the example video); if an integer, it will return an image with that many
                   faces in it (see medusa/data/example_data/images folder); if a list (or tuple),
                   it will return a list of images with the number of faces specified in the list
   :type n_faces: int, list, None
   :param load_numpy: Whether to load it as a numpy array
   :type load_numpy: bool
   :param load_torch: Whether to load it as a torch array
   :type load_torch: bool
   :param device: Either 'cuda' or 'cpu'; ignored when ``load_torch`` is False
   :type device: str

   :returns: **img** -- A path or a 3D numpy array/torch Tensor of shape frame width x height x 3 (RGB)
   :rtype: pathlib.Path, np.ndarray, torch.Tensor

   .. rubric:: Notes

   If both ``load_numpy`` and ``load_torch`` are False, then just
   a ``pathlib.Path`` object is returned.

   .. rubric:: Examples

   >>> # Load path to example image frame
   >>> img = get_example_image()
   >>> img.is_file()
   True
   >>> # Load file as numpy array
   >>> img = get_example_image(load_numpy=True)
   >>> img.shape
   (384, 480, 3)


.. py:function:: get_example_video(n_faces=None, return_videoloader=False, **kwargs)

   Retrieves the path to an example video file.

   :param n_faces: If None, it will return the default (example) video; if an integer, it will
                   return an image with that many faces in it (see medusa/data/example_data/videos folder)
   :type n_faces: int, None
   :param return_videoloader: Returns the video as a ``VideoLoader`` object
   :type return_videoloader: bool
   :param kwargs: Extra parameters passed to the ``VideoLoader`` initialization;
                  ignored when ``return_videoloader`` is False
   :type kwargs: dict

   :returns: **path** -- A Path object pointing towards the example
             video file or a ``VideoLoader`` object
   :rtype: pathlib.Path, VideoLoader

   .. rubric:: Examples

   Get just the file path (as a ``pathlib.Path`` object)

   >>> path = get_example_video()
   >>> path.is_file()
   True

   Get it as a ``VideoLoader`` object to quickly get batches of images already
   loaded on and formatted for GPU:

   >>> vid = get_example_video(return_videoloader=True, batch_size=32)
   >>> # We can loop over `vid` or just get a single batch, as below:
   >>> img_batch = next(vid)
   >>> img_batch.shape
   torch.Size([32, 384, 480, 3])


.. py:function:: get_example_data4d(n_faces=None, load=False, model='mediapipe', device=DEVICE)

   Retrieves an example hdf5 file with reconstructed 4D data from the
   example video.

   :param n_faces: If None, it will return the reconstruction from the default (example) video; if
                   an integer, it will return the recon data from the video with that many faces in
                   it (see medusa/data/example_data/videos folder)
   :type n_faces: int, None
   :param load: Whether to return the hdf5 file loaded in memory (``True``)
                or to just return the path to the file
   :type load: bool
   :param model: Model used to reconstruct the data; either 'mediapipe' or
                 'emoca'
   :type model: str

   :returns: When ``load`` is ``True``, returns either a ``MediapipeData``
             or a ``FlameData`` object, otherwise a string or ``pathlib.Path``
             object to the file
   :rtype: MediapipeData, FlameData, str, Path

   .. rubric:: Examples

   >>> path = get_example_data4d(load=False, as_path=True)
   >>> path.is_file()
   True

   # Get hdf5 file already loaded in memory
   >>> data = get_example_data4d(load=True, model='mediapipe')
   >>> data.recon_model
   'mediapipe'
   >>> data.v.shape  # check out reconstructed vertices
   (232, 468, 3)


