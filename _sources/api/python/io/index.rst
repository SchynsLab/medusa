:py:mod:`medusa.io`
===================

.. py:module:: medusa.io

.. autoapi-nested-parse::

   Module with functionality (mostly) for working with video data.

   The ``VideoLoader`` class allows for easy looping over frames of a video
   file, which is used in the reconstruction process (e.g., in the
   ``videorecon`` function).



Module Contents
---------------

.. py:class:: VideoLoader(video_path, batch_size=32, channels_first=False, device=DEVICE, crop=None, **kwargs)



   Contains (meta)data and functionality associated with video files (mp4
   files only currently).

   :param path: Path to mp4 file
   :type path: str, Path
   :param batch_size: Batch size to use when loading frames
   :type batch_size: int
   :param channels_first: Whether to return a B x 3 x H x W tensor (if ``True``) or
                          a B x H x W x 3 tensor (if ``False``)
   :type channels_first: bool
   :param device: Either 'cpu' or 'cuda'
   :type device: str
   :param \*\*kwargs: Extra keyword arguments passed to the initialization of the parent class

   .. py:method:: get_metadata()

      Returns all (meta)data needed for initialization of a Data object.

      :returns: * *A dictionary with keys "img_size" (image size of frames), "n_img" (total number*
                * *of frames), and "fps" (frames-per-second)*


   .. py:method:: close()

      Closes the opencv videoloader in the underlying pytorch Dataset.



.. py:class:: VideoDataset(video_path, device=DEVICE)



   A pytorch Dataset class based on loading frames from a single video.

   :param video: A video file (any format that pyav can handle)
   :type video: pathlib.Path, str
   :param device: Either 'cuda' (for GPU) or 'cpu'
   :type device: str

   .. py:method:: close()

      Closes the pyav videoreader and free up memory.



.. py:class:: VideoWriter(path, fps, codec='libx264', pix_fmt='yuv420p', size=None)

   A PyAV based images-to-video writer.

   :param path: Output path (including extension)
   :type path: str, Path
   :param fps: Frames per second of output video; if float, it's rounded
               and cast to int
   :type fps: float, int
   :param codec: Video codec to use (e.g., 'mpeg4', 'libx264', 'h264')
   :type codec: str
   :param pix_fmt: Pixel format; should be compatible with codec
   :type pix_fmt: str
   :param size: Desired output size of video (if ``None``, wil be set the first time a frame
                is written)
   :type size: tuple[int]

   .. py:method:: write(imgs)

      Writes one or more images to the video stream.

      :param imgs: A torch tensor or numpy array with image data; can be
                   a single image or batch of images
      :type imgs: array_like


   .. py:method:: close()

      Closes the video stream.



.. py:function:: load_inputs(inputs, load_as='torch', channels_first=True, with_batch_dim=True, dtype='float32', device=DEVICE)

   Generic image loader function, which also performs some basic
   preprocessing and checks. Is used internally for detection, crop, and
   reconstruction models.

   :param inputs: String or ``Path`` to a single image or an iterable (list, tuple) with
                  multiple image paths, or a numpy array or torch Tensor with already
                  loaded images (in which the first dimension represents the number of images)
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
   :rtype: np.ndarray, torch.tensor

   .. rubric:: Examples

   Load a single image as a torch Tensor:
   >>> from medusa.data import get_example_image
   >>> path = get_example_image()
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


.. py:function:: download_file(url, f_out, data=None, verify=True, overwrite=False, cmd_type='post')

   Downloads a file using requests. Used internally to download external
   data.

   :param url: URL of file to download
   :type url: str
   :param f_out: Where to save the downloaded file
   :type f_out: Path
   :param data: Extra data to pass to post request
   :type data: dict
   :param verify: Whether to verify the request
   :type verify: bool
   :param overwrite: Whether to overwrite the file when it already exists
   :type overwrite: bool
   :param cmd_type: Either 'get' or 'post'
   :type cmd_type: str


.. py:function:: load_obj(f, device=None)

   Loads data from obj file, based on the DECA implementation, which in
   turn is based on the pytorch3d implementation.

   :param f: Filename of object file
   :type f: str, Path
   :param device: If None, returns numpy arrays. Otherwise, returns torch tensors on this device
   :type device: str, None

   :returns: **out** -- Dictionary with outputs (keys: 'v', 'tris', 'vt', 'tris_uv')
   :rtype: dict


.. py:function:: save_obj(f, data)

   Saves data to an obj file, based on the implementation from PRNet.

   :param f: Path to save file to
   :type f: str, Path
   :param data: Dictionary with 3D mesh data, with keys 'v', 'tris', and optionally 'vt'
   :type data: dict


