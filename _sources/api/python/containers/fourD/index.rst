:py:mod:`medusa.containers.fourD`
=================================

.. py:module:: medusa.containers.fourD

.. autoapi-nested-parse::

   Module with core 4D functionality of the ``medusa`` package, most
   importantly the ``Data4D`` class, which stores reconstructed data from videos
   and other (meta)data needed to further process, analyze, and visualize it.

   The data can be saved to disk as a `HDF5
   <https://www.hdfgroup.org/solutions/hdf5/>`_ file (using `h5py
   <http://www.h5py.org/>`_) with the ``save`` method and loaded from disk
   using the ``load`` classmethod.



Module Contents
---------------

.. py:class:: Data4D(v, mat, tris=None, img_idx=None, face_idx=None, video_metadata=None, cam_mat=None, space='world', device=DEVICE)

   Data class which stores reconstruction data and provides methods to
   preprocess/manipulate them.

   :param v: Numpy array or torch tensor of shape T (time points) x nV (no. vertices) x 3 (x/y/z)
   :type v: np.ndarray, torch.tensor
   :param tris: Integer numpy array or torch tensor of shape n_t (no. of triangles) x 3 (vertices per triangle)
   :type tris: ndarray, torch.tensor
   :param mat: Numpy array of shape T (time points) x 4 x 4 (affine matrix) representing
               the 'world' (or 'model') matrix for each time point
   :type mat: ndarray
   :param face_idx: Integer numpy array with indices that map vertices to distinct faces
   :type face_idx: ndarray
   :param cam_mat: Numpy array of shape 4x4 (affine matrix) representing the camera matrix
   :type cam_mat: ndarray
   :param space: The space the vertices are currently in; can be either 'local' or 'world'
   :type space: str

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


   .. py:method:: apply_vertex_mask(name)

      Applies a mask to the vertices (and triangles).

      :param name: Name of masks (one of 'face', 'lips', 'neck', 'nose', 'boundary', 'forehead',
                   'scalp')
      :type name: str


   .. py:method:: from_video(path, **kwargs)
      :staticmethod:

      Utility method to directly initialize a ``Data4D`` object by calling
      the ``videorecon`` function.

      :param path: Path to video that will be reconstructed
      :type path: str, pathlib.Path
      :param \*\*kwargs: Keyword arguments passed to ``videorecon``

      :returns: **data** -- A Data4D object
      :rtype: Data4D


   .. py:method:: load(path, device=None)
      :classmethod:

      Loads an HDF5 file from disk, parses its contents, and creates the
      initialization parameters necessary to initialize a ``*Data`` object.

      :param path: A path towards an HDF5 file data reconstructed by Medusa
      :type path: str, pathlib.Path

      :rtype: An initialized Data4D object


   .. py:method:: to_local()

      Converts the data to local space.


   .. py:method:: to_world()

      Converts the data to world space.


   .. py:method:: project_to_68_landmarks()

      Projects to 68 landmark set.

      :rtype: v_proj


   .. py:method:: get_face(index, pad_missing=True)

      Get the data from a particular face in the reconstruction.

      :param index: Integer index corresponding to the face
      :type index: int


   .. py:method:: decompose_mats(to_df=True)

      Decomponses a time series (of length T) 4x4 affine matrices to a
      numpy array (or pandas ``DataFrame``) with a time series of T x 12
      affine parameters (translation XYZ, rotation XYZ, scale XYZ, shear
      XYZ).

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

      Converts a sequence of global (affine) motion parameters into a
      sequence of 4x4 affine matrices and updates the ``.mat`` attribute.
      Essentially does the opposite of the ``decompose_mats`` method.

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


   .. py:method:: filter_faces(present_threshold=0.1)

      Filters the reconstructed faces by the proportion of frames they are
      present in.

      :param present_threshold: Lower bound on proportion present
      :type present_threshold: float



