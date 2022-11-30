:py:mod:`medusa.core.fourD`
===========================

.. py:module:: medusa.core.fourD

.. autoapi-nested-parse::

   Module with core 4D functionality of the ``medusa`` package, most importantly the
   ``*4D`` classes. The ``Base4D`` class defines a template class from which
   model-specific classes (such as ``Flame4D``) inherit. Objects initialized from these
   classes store reconstructed data from videos and other (meta)data needed to further
   process, analyze, and visualize it.

   The reconstructed data from each model supported by ``medusa`` is stored in an object from
   a specific class which inherits from ``Base4D``. For example, reconstructed data from
   `mediapipe <https://google.github.io/mediapipe/solutions/face_mesh.html>`_ is stored
   in using the ``Mediapipe4D`` class. Other classes include the ``Flame4D`` for reconstructions
   from models using the `FLAME topology <https://flame.is.tue.mpg.de/>`_ (such as
   `EMOCA <https://emoca.is.tue.mpg.de/>`_).

   The data can be saved to disk as a `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
   file (using `h5py <http://www.h5py.org/>`_) with the ``save`` method and loaded from
   disk using the ``load`` (static)method.



Module Contents
---------------

.. py:class:: Base4D(v, f, mat=None, cam_mat=None, frame_t=None, sf=None, img_size=(512, 512), recon_model=None, space='world', path=None, loglevel='INFO')

   Base Data class with attributes and methods common to all 4D data classes
   (such as ``Flame4D``, ``Mediapipe4D``, etc.).

   Warning: objects should never be initialized with this class directly,
   only when calling ``super().__init__()`` from the subclass (like ``Flame4D``). Note,
   though, that the initialization parameters are the same for every class that
   inherits from ``Base4D``.

   :param v: Numpy array of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
   :type v: ndarray
   :param f: Integer numpy array of shape nF (no. faces) x 3 (vertices per face); can be
             `None` if working with landmarks/vertices only
   :type f: ndarray
   :param mat: Numpy array of shape T (time points) x 4 x 4 (affine matrix) representing
               the 'world' (or 'model') matrix for each time point
   :type mat: ndarray
   :param cam_mat: Numpy array of shape 4x4 (affine matrix) representing the camera matrix
   :type cam_mat: ndarray
   :param frame_t: Numpy array of length T (time points) with "frame times", i.e.,
                   onset of each frame (in seconds) from the video
   :type frame_t: ndarray
   :param sf: Sampling frequency of video
   :type sf: int, float
   :param recon_model: Name of reconstruction model
   :type recon_model: str
   :param space: The space the vertices are currently in; can be either 'local' or 'world'
   :type space: str
   :param path: Path where the data is saved; if initializing a new object (rather than
                loading one from disk), this should be `None`
   :type path: str
   :param loglevel: Logging level of current logger
   :type loglevel: int

   .. py:method:: project_to_68_landmarks()

      Projects to 68 landmark set.


   .. py:method:: decompose_mats(to_df=True)

      Decomponses a time series (of length T) 4x4 affine matrices to a numpy array
      (or pandas ``DataFrame``) with a time series of T x 12 affine parameters
      (translation XYZ, rotation XYZ, scale XYZ, shear XYZ).

      :param to_df: Whether to return the parameters as a pandas ``DataFrame`` or
                    not (in which case it's returned as a numpy array)
      :type to_df: bool

      :returns: **params** -- Either a ``DataFrame`` or numpy array, depending on the ``to_df`` parameter
      :rtype: pd.DataFrame, np.ndarray

      .. rubric:: Examples

      Convert the sequences of affine matrices to a 2D numpy array:

      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True, model="mediapipe")
      >>> params = data.decompose_mats(to_df=False)
      >>> params.shape
      (232, 12)


   .. py:method:: compose_mats(params)

      Converts a sequence of global (affine) motion parameters into a sequence
      of 4x4 affine matrices and updates the ``.mat`` attribute. Essentially
      does the opposite of the ``decompose_mats`` method.

      :param params: A 2D numpy array of shape T (time points) x 12
      :type params: np.ndarray

      .. rubric:: Examples

      Convert the sequences of affine matrices to a 2D numpy array and uses the
      ``compose_mats`` function to reverse it.

      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True, model="mediapipe")
      >>> orig_mats = data.mat.copy()
      >>> params = data.decompose_mats(to_df=False)
      >>> data.compose_mats(params)
      >>> np.testing.assert_array_almost_equal(orig_mats, data.mat)  # passes!


   .. py:method:: save_obj(idx, path)


   .. py:method:: save(path, compression_level=9)

      Saves (meta)data to disk as an HDF5 file.

      :param path: Path to save the data to
      :type path: str
      :param compression_level: Level of compression (higher = more compression, but slower; max = 9)
      :type compression_level: int

      .. rubric:: Examples

      Save data to disk:

      >>> import os
      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True, model="mediapipe")
      >>> data.save('./my_data.h5')
      >>> os.remove('./my_data.h5')  # clean up


   .. py:method:: load(path)
      :staticmethod:

      Loads an HDF5 file from disk, parses its contents, and creates the
      initialization parameters necessary to initialize a ``*Data`` object. It
      does not return a ``*Data`` object itself; only a dictionary with the parameters.

      Important: it is probably better to call the ``load`` method from a specific
      data class (e.g., ``Mediapipe4D``) than the ``load`` method from the
      ``Base4D`` class.

      :param path: A path towards an HDF5 file data reconstructed by Medusa
      :type path: str, pathlib.Path

      :returns: **init_kwargs** -- Parameters necessary to initialize a ``*4D`` object.
      :rtype: dict

      .. rubric:: Examples

      Get Mediapipe reconstruction data and initialize a ``Mediapipe4D`` object.
      Note that it's easier to just call the ``load`` classmethod from the
      ``Mediapipe4D`` class directly, i.e., ``Mediapipe4D.load(path)``.

      >>> from medusa.data import get_example_h5
      >>> from medusa.core import Mediapipe4D
      >>> path = get_example_h5(load=False, model="mediapipe")
      >>> init_kwargs = Base4D.load(path)
      >>> data = Mediapipe4D(**init_kwargs)


   .. py:method:: render_video(f_out, renderer, video=None, scale=None, n_frames=None, alpha=None, overlay=None)

      Renders the sequence of 3D meshes as a video. It is assumed that this
      method is only called from a child class (e.g., ``Mediapipe4D``).

      :param f_out: Filename of output
      :type f_out: str
      :param renderer: The renderer object
      :type renderer: ``medusa.render.Renderer``
      :param video: Path to video, in order to render face on top of original video frames
      :type video: str
      :param scale: A scaling factor of the resulting video; 0.25 means 25% of original size
      :type scale: float
      :param n_frames: Number of frames to render; e.g., ``10`` means "render only the first
                       10 frames of the video"; nice for debugging. If ``None`` (default), all
                       frames are rendered
      :type n_frames: int
      :param alpha: Alpha (transparency) level of the rendered face; lower = more transparent;
                    minimum = 0 (invisible), maximum = 1 (fully opaque)
      :type alpha: float


   .. py:method:: copy()


   .. py:method:: __len__()

      Returns the number of time points of the reconstructed vertices (i.e.,
      the number of reconstructed frames from the video.


   .. py:method:: __getitem__(idx)

      Returns the vertices at a particular time point (``idx``).

      :param idx: Index into the time dimension of the data
      :type idx: int


   .. py:method:: __setitem__(idx, v)

      Replace the vertices at time point ``idx`` with ``v``.

      :param idx: Index into the time dimension of the data
      :type idx: int
      :param v: Numpy array with vertices of shape ``nV`` (number of verts) x 3 (XYZ)
      :type v: np.ndarray



.. py:class:: Flame4D(*args, **kwargs)

   Bases: :py:obj:`Base4D`

   4D data class specific to reconstructions from models based on the FLAME
   topology.

   Warning: we recommend against initializing a ``Flame4D`` object directly
   (i.e., through the ``__init__`` class constructor). Instead, use the high-level
   ``videorecon`` function, which returns a ``Flame4D`` object. Or, if you
   are loading data from disk, use the ``load`` classmethod (see examples)

   :param \*args: Positional (non-keyword) arguments passed to the ``Base4D`` constructor
   :type \*args: iterable
   :param \*\*kwargs: Keyword arguments passed to the ``Base4D`` constructor
   :type \*\*kwargs: dict

   .. rubric:: Examples

   We recommend creating ``Flame4D`` objects by loading the corresponding
   HDF5 file from disk (see ``load`` docstring).

   .. py:method:: load(path)
      :classmethod:

      Loads existing data (stored as an HDF5 file) from disk and uses it to
      instantiate a ``Flame4D`` object.

      :param path: A path to an HDF5 file with data from a Flame-based reconstruction model
      :type path: str, pathlib.Path

      :rtype: An ``Flame4D`` object

      .. rubric:: Examples

      Load data from a ``mediapipe`` reconstruction:

      >>> from medusa.data import get_example_h5
      >>> path_to_h5 = get_example_h5(load=False)
      >>> data = Flame4D.load(path_to_h5)
      >>> type(data)
      <class 'medusa.core.fourD.Flame4D'>


   .. py:method:: render_video(f_out, smooth=False, wireframe=False, **kwargs)

      Renders a video from the 4D reconstruction.

      :param f_out: Path to save the video to
      :type f_out: str, pathlib.Path
      :param smooth: Whether to render a smooth face (using smooth shading) or not (using flat
                     shading)
      :type smooth: bool
      :param wireframe: Whether to render a wireframe instead of an opaque face (if ``True``, the
                        ``smooth`` parameter is ignored)
      :type wireframe: bool
      :param kwargs: Additional keyword arguments passed to the ``Base4D.render_video`` method
      :type kwargs: dict

      .. rubric:: Examples

      Render a video



.. py:class:: Mediapipe4D(*args, **kwargs)

   Bases: :py:obj:`Base4D`

   4D data class specific to reconstructions from the Mediapipe model.

   Warning: we recommend against initializing a ``Mediapipe4D`` object directly
   (i.e., through the ``__init__`` class constructor). Instead, use the high-level
   ``videorecon`` function, which returns a ``Mediapipe4D`` object. Or, if you
   are loading data from disk, use the ``load`` classmethod (see examples)

   :param \*args: Positional (non-keyword) arguments passed to the ``Base4D`` constructor
   :type \*args: iterable
   :param \*\*kwargs: Keyword arguments passed to the ``Base4D`` constructor
   :type \*\*kwargs: dict

   .. rubric:: Examples

   We recommend creating ``Mediapipe4D`` objects by loading the corresponding
   HDF5 file from disk (see ``load`` docstring).

   .. py:method:: load(path)
      :classmethod:

      Loads Mediapipe data from a HDF5 file and returns a ``Mediapipe4D``
      object.

      :param path: Path to HDF5 file with Mediapipe data
      :type path: str, pathlib.Path

      :rtype: A ``Mediapipe4D`` object

      .. rubric:: Examples

      The ``load`` classmethod is the recommended way to initialize a ``Mediapipe4D``
      object with already reconstructed data:

      >>> from medusa.data import get_example_h5
      >>> path = get_example_h5()
      >>> mp_data = Mediapipe4D.load(path)

      If the data is not reconstructed yet, use the ``videorecon`` function to create
      such an object:

      >>> from medusa.recon import videorecon
      >>> from medusa.data import get_example_video
      >>> path = get_example_video()
      >>> mp_data = videorecon(path, recon_model='mediapipe')


   .. py:method:: render_video(f_out, smooth=False, wireframe=False, **kwargs)

      Renders a video of the reconstructed vertices.

      Note: the extension of the ``f_out`` parameter (at the moment only ".mp4")
      determines the format of the rendered video.

      :param f_out: Path where the video should be saved
      :type f_out: str, pathlib.Path
      :param smooth: Whether to render a smooth mesh or not (ignored when ``wireframe=True``)
      :type smooth: bool
      :param wireframe: Whether to render wireframe instead of the full mesh
      :type wireframe: bool
      :param \*\*kwargs: Keyword arguments passed to the ``render_video`` method from ``Base4D``;
                         options include ``video``, ``scaling``, ``n_frames``, and ``alpha``
      :type \*\*kwargs: dict

      .. rubric:: Examples

      Rendering a video with just the wireframe:

      >>> from pathlib import Path
      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True)
      >>> f_out = Path('./example_vid_recon.mp4')
      >>> data.render_video(f_out, wireframe=True)
      >>> f_out.is_file()
      True

      Rendering an MP4 video with a smooth mesh on top of the original video:

      >>> from medusa.data import get_example_video
      >>> vid = get_example_video()
      >>> data = get_example_h5(load=True)
      >>> f_out = Path('./example_vid_recon.mp4')
      >>> data.render_video(f_out, smooth=True, video=vid)
      >>> f_out.is_file()
      True



