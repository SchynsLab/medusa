:py:mod:`medusa.io`
===================

.. py:module:: medusa.io

.. autoapi-nested-parse::

   Module with functionality (mostly) for working with video data.
   The ``VideoLoader`` class allows for easy looping over frames of a video file,
   which is used in the reconstruction process (e.g., in the ``videorecon`` function).



Module Contents
---------------

.. py:class:: VideoLoader(path, rescale_factor=None, n_preload=512, device=DEVICE, batch_size=32, loglevel='INFO', **kwargs)

   Bases: :py:obj:`torch.utils.data.DataLoader`

   " Contains (meta)data and functionality associated
   with video files (mp4 files only currently).

   :param path: Path to mp4 file
   :type path: str, Path
   :param rescale_factor: Rescale factor of video frames (e.g., 0.25 means scale each dimension to 25% of original);
                          if ``None`` (default), the image is not resized
   :type rescale_factor: float
   :param n_preload: Number of video frames to preload before batching
   :type n_preload: int
   :param loglevel: Logging level (e.g., 'INFO' or 'WARNING')
   :type loglevel: str

   :raises ValueError: If `n_preload` is not a multiple of `batch_size`

   .. py:method:: get_metadata()

      Returns all (meta)data needed for initialization
      of a Data object.


   .. py:method:: close()

      Closes the opencv videoloader in the underlying pytorch Dataset.


   .. py:method:: __len__()

      Utility function to easily access number of video frames.


   .. py:method:: __next__()

      Return the next batch of the dataloader.



.. py:class:: VideoDataset(video, rescale_factor=None, n_preload=512, device='cuda')

   Bases: :py:obj:`torch.utils.data.Dataset`

   A pytorch Dataset class based on loading frames from a single video.

   :param video: A video file (any format that cv2 can handle)
   :type video: pathlib.Path, str
   :param rescale_factor: Factor with which to rescale the input image (for speed)
   :type rescale_factor: float
   :param n_preload: How many frames to preload before batching; higher values will
                     take up more RAM, but result in faster loading
   :type n_preload: int
   :param device: Either 'cuda' (for GPU) or 'cpu'
   :type device: str

   .. py:method:: __len__()


   .. py:method:: __getitem__(i)


   .. py:method:: close()

      Closes the cv2 videoreader and free up memory.



.. py:function:: load_h5(path)

   Convenience function to load a hdf5 file and immediately initialize the correct
   data class.

   :param path: Path to an HDF5 file
   :type path: str

   :returns: **data** -- An object with a class derived from ``data.BaseData``
             (like ``MediapipeData``, or ``FlameData``)
   :rtype: ``data.BaseData`` subclass

   .. rubric:: Examples

   Load in HDF5 data reconstructed by Mediapipe:

   >>> from medusa.data import get_example_h5
   >>> path = get_example_h5(load=False)
   >>> data = load_h5(path)


.. py:function:: load_inputs(inputs, load_as='torch', channels_first=True, with_batch_dim=True, dtype='float32', device=DEVICE)

   Generic image loader function, which also performs some basic
   preprocessing and checks. Is used internally for crop models and
   reconstruction models.

   :param inputs: String or Path to a single image or an iterable (list, tuple) with
                  multiple image paths, or a numpy array or torch Tensor with already
                  loaded images
   :type inputs: str, Path, iterable, array_like
   :param load_as: Either 'torch' (returns torch Tensor) or 'numpy' (returns numpy ndarray)
   :type load_as: str
   :param to_bgr: Whether the color channel is ordered BGR (True) or RGB (False); only
                  works when inputs are image path(s)
   :type to_bgr: bool
   :param channels_first: Whether the data is ordered as (batch_size, 3, h, w) (True) or
                          (batch_size, h, w, 3) (False)
   :type channels_first: bool
   :param with_batch_dim: Whether a singleton batch dimension should be added if there's only
                          a single image
   :type with_batch_dim: bool
   :param dtype: Data type to be used for loaded images (e.g., 'float32', 'float64', 'uint8')
   :type dtype: str
   :param device: Either 'cuda' (for GPU) or 'cpu'; ignored when ``load_as='numpy'``
   :type device: str

   :returns: **imgs** -- Images loaded in memory; object depends on the ``load_as`` parameter
   :rtype: np.ndarray, torch.Tensor

   .. rubric:: Examples

   Load a single image as a torch Tensor:
   >>> from medusa.data import get_example_frame
   >>> path = get_example_frame()
   >>> img = load_inputs(path, device='cpu')
   >>> img.shape
   torch.Size([1, 3, 384, 480])

   Or as a numpy array (without batch dimension):

   >>> img = load_inputs(path, load_as='numpy', with_batch_dim=False)
   >>> img.shape
   (3, 384, 480)

   Putting the channel dimension last:

   >>> img = load_inputs(path, load_as='numpy', channels_first=False)
   >>> img.shape
   (1, 384, 480, 3)

   Setting the data type to uint8 instead of float32:

   >>> img = load_inputs(path, load_as='torch', dtype='uint8', device='cpu')
   >>> img.dtype
   torch.uint8

   Loading in a list of images:

   >>> img = load_inputs([path, path], load_as='numpy')
   >>> img.shape
   (2, 3, 384, 480)


.. py:function:: save_obj(v, f, f_out)


.. py:function:: download_file(url, f_out, data=None, verify=True, overwrite=False, cmd_type='post')


