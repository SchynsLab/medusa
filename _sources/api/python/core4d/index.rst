:py:mod:`medusa.core4d`
=======================

.. py:module:: medusa.core4d

.. autoapi-nested-parse::

   Module with core 4D functionality of the ``medusa`` package, most importantly the
   ``*4D`` classes. The ``Base4D`` class defines a template class from which
   model-specific classes (such as ``Flame4D``) inherit. Objects initialized from these
   classes store reconstructed data from videos and other (meta)data needed to further
   process, analyze, and visualize it.

   The reconstructed data from each model supported by ``medusa`` is stored in an object from
   a specific class which inherits from ``Base4D``. For example, reconstructed data from
   `mediapipe <https://google.github.io/mediapipe/solutions/face_mesh.html>`_ is stored
   in using the ``Mediapipe4D`` class. Other classes include the ``Fan4D`` for
   reconstructions from `FAN <https://github.com/1adrianb/face-alignment>`_ and
   ``Flame4D`` for reconstructions from models using the `FLAME topology <https://flame.is.tue.mpg.de/>`_
   (such as `EMOCA <https://emoca.is.tue.mpg.de/>`_).

   The data can be saved to disk as a `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
   file (using `h5py <http://www.h5py.org/>`_) with the ``save`` method and loaded from
   disk using the ``load`` (static)method.



Module Contents
---------------

.. py:class:: Base4D(v, f=None, mat=None, cam_mat=None, frame_t=None, events=None, sf=None, img_size=(512, 512), recon_model_name=None, space='world', path=None, loglevel=20)

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
   :param events: A Pandas DataFrame with `N` rows corresponding to `N` separate trials
                  and at least two columns: 'onset' and 'trial_type' (optional: 'duration')
   :type events: pd.DataFrame
   :param sf: Sampling frequency of video
   :type sf: int, float
   :param recon_model_name: Name of reconstruction model
   :type recon_model_name: str
   :param space: The space the vertices are currently in; can be either 'local' or 'world'
   :type space: str
   :param path: Path where the data is saved; if initializing a new object (rather than
                loading one from disk), this should be `None`
   :type path: str
   :param loglevel: Logging level of current logger
   :type loglevel: int

   .. py:method:: mats2params(self, to_df=True)

      Transforms a time series (of length T) 4x4 affine matrices to a
      pandas DataFrame with a time series of T x 12 affine parameters
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
      >>> params = data.mats2params(to_df=False)
      >>> params.shape
      (232, 12)


   .. py:method:: params2mats(self, params)

      Converts a sequence of global (affine) motion parameters into a sequence
      of 4x4 affine matrices and updates the ``.mat`` attribute. Essentially
      does the opposite of the ``mats2params`` method.

      :param params: A 2D numpy array of shape T (time points) x 12
      :type params: np.ndarray

      .. rubric:: Examples

      Convert the sequences of affine matrices to a 2D numpy array and uses the
      ``params2mats`` function to reverse it.

      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True, model="mediapipe")
      >>> orig_mats = data.mat.copy()
      >>> params = data.mats2params(to_df=False)
      >>> data.params2mats(params)
      >>> np.testing.assert_array_almost_equal(orig_mats, data.mat)  # passes!


   .. py:method:: save(self, path, compression_level=9)

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
      >>> from medusa.core4d import Mediapipe4D
      >>> path = get_example_h5(load=False, model="mediapipe")
      >>> init_kwargs = Base4D.load(path)
      >>> data = Mediapipe4D(**init_kwargs)


   .. py:method:: to_mne_rawarray(self)

      Creates an MNE `RawArray` object from the vertices (`v`).

      .. rubric:: Examples

      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True)
      >>> rawarray = data.to_mne_rawarray()


   .. py:method:: render_video(self, f_out, renderer, video=None, scaling=None, n_frames=None, alpha=None)

      Renders the sequence of 3D meshes as a video. It is assumed that this
      method is only called from a child class (e.g., ``Mediapipe4D``).

      :param f_out: Filename of output
      :type f_out: str
      :param renderer: The renderer object
      :type renderer: ``medusa.render.Renderer``
      :param video: Path to video, in order to render face on top of original video frames
      :type video: str
      :param scaling: A scaling factor of the resulting video; 0.25 means 25% of original size
      :type scaling: float
      :param n_frames: Number of frames to render; e.g., ``10`` means "render only the first
                       10 frames of the video"; nice for debugging. If ``None`` (default), all
                       frames are rendered
      :type n_frames: int
      :param alpha: Alpha (transparency) level of the rendered face; lower = more transparent;
                    minimum = 0 (invisible), maximum = 1 (fully opaque)
      :type alpha: float


   .. py:method:: plot_data(self, f_out, plot_motion=True, plot_pca=True, n_pca=3)

      Creates a plot of the motion (rotation & translation) parameters
      over time and the first `n_pca` PCA components of the
      reconstructed vertices. For FLAME and Mediapipe estimates, these parameters are
      relative to the canonical model, so the estimates are plotted relative
      to the value of the first frame.

      :param f_out: Where to save the plot to (a png file)
      :type f_out: str, Path
      :param plot_motion: Whether to plot the motion parameters
      :type plot_motion: bool
      :param plot_pca: Whether to plot the `n_pca` PCA-transformed traces of the data (`self.v`)
      :type plot_pca: bool
      :param n_pca: How many PCA components to plot
      :type n_pca: int

      .. rubric:: Examples

      >>> import os
      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True)
      >>> data.plot_data('./example_plot.png')
      >>> os.remove('./example_plot.png')


   .. py:method:: __len__(self)

      Returns the number of time points of the reconstructed vertices (i.e.,
      the number of reconstructed frames from the video.


   .. py:method:: __getitem__(self, idx)

      Returns the vertices at a particular time point (``idx``).

      :param idx: Index into the time dimension of the data
      :type idx: int


   .. py:method:: __setitem__(self, idx, v)

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

   .. py:method:: load(cls, path)
      :classmethod:

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
      >>> from medusa.core4d import Mediapipe4D
      >>> path = get_example_h5(load=False, model="mediapipe")
      >>> init_kwargs = Base4D.load(path)
      >>> data = Mediapipe4D(**init_kwargs)


   .. py:method:: render_video(self, f_out, smooth=False, wireframe=False, **kwargs)

      Renders the sequence of 3D meshes as a video. It is assumed that this
      method is only called from a child class (e.g., ``Mediapipe4D``).

      :param f_out: Filename of output
      :type f_out: str
      :param renderer: The renderer object
      :type renderer: ``medusa.render.Renderer``
      :param video: Path to video, in order to render face on top of original video frames
      :type video: str
      :param scaling: A scaling factor of the resulting video; 0.25 means 25% of original size
      :type scaling: float
      :param n_frames: Number of frames to render; e.g., ``10`` means "render only the first
                       10 frames of the video"; nice for debugging. If ``None`` (default), all
                       frames are rendered
      :type n_frames: int
      :param alpha: Alpha (transparency) level of the rendered face; lower = more transparent;
                    minimum = 0 (invisible), maximum = 1 (fully opaque)
      :type alpha: float



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

   .. py:method:: load(cls, path)
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

      >>> from medusa.preproc import videorecon
      >>> from medusa.data import get_example_video
      >>> path = get_example_video()
      >>> mp_data = videorecon(path, recon_model_name='mediapipe')


   .. py:method:: render_video(self, f_out, smooth=False, wireframe=False, **kwargs)

      Renders a video of the reconstructed vertices.

      Note: the extension of the ``f_out`` parameter (e.g., ".gif" or ".mp4")
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

      Rendering a GIF with just the wireframe:

      >>> from pathlib import Path
      >>> from medusa.data import get_example_h5
      >>> data = get_example_h5(load=True)
      >>> f_out = Path('./example_vid_recon.gif')
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



.. py:class:: Fan4D(*args, **kwargs)

   Bases: :py:obj:`Base4D`

   Data class specific to reconstructions from the FAN (3D) model.

   Warning: we recommend against initializing a ``Fan4D`` object directly
   (i.e., through the ``__init__`` class constructor). Instead, use the high-level
   ``videorecon`` function, which returns a ``Fan4D`` object. Or, if you
   are loading data from disk, use the ``load`` classmethod (see examples)

   :param \*args: Positional (non-keyword) arguments passed to the ``Base4D`` constructor
   :type \*args: iterable
   :param \*\*kwargs: Keyword arguments passed to the ``Base4D`` constructor
   :type \*\*kwargs: dict

   .. rubric:: Examples

   We recommend creating ``Fan4D`` objects by loading the corresponding
   HDF5 file from disk (see ``load`` docstring).

   .. py:method:: load(cls, path)
      :classmethod:

      Loads FAN data from a HDF5 file and returns a ``Fan4D`` object.

      :param path: Path to HDF5 file with FAN data
      :type path: str, pathlib.Path

      :rtype: A ``Fan4D`` object

      .. rubric:: Examples

      If the data is not reconstructed yet, use the ``videorecon`` function to create
      such an object:

      >>> from medusa.preproc import videorecon
      >>> from medusa.data import get_example_video
      >>> path = get_example_video()
      >>> fan_data = videorecon(path, recon_model_name='fan', device='cpu')


   .. py:method:: render_video(self, f_out, video=None)

      Renders a video of the reconstructed vertices.

      Note: the extension of the ``f_out`` parameter (e.g., ".gif" or ".mp4")
      determines the format of the rendered video.

      :param f_out: Path where the video should be saved
      :type f_out: str, pathlib.Path
      :param video: Path to video, if you want to render the face on top of the original video;
                    default is ``None`` (i.e., do not render on top of video)
      :type video: str, pathlib.Path

      .. rubric:: Examples

      Rendering a GIF with wireframe (only possibility) on top of video:

      >>> from pathlib import Path
      >>> from medusa.data import get_example_video
      >>> from medusa.data import get_example_h5
      >>> vid = get_example_video()
      >>> data = get_example_h5(load=True, model='fan')
      >>> f_out = Path('./example_vid_recon.gif')
      >>> data.render_video(f_out, video=vid)
      >>> f_out.is_file()
      True



