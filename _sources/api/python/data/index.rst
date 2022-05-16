:py:mod:`medusa.data`
=====================

.. py:module:: medusa.data


Module Contents
---------------

.. py:data:: logger
   

   

.. py:data:: here
   

   

.. py:class:: BaseData(v, f=None, mat=None, cam_mat=None, frame_t=None, events=None, sf=None, img_size=None, recon_model_name=None, space='world', path=None)

   Base Data class with attributes and methods common to all Data
   classes (such as FlameData, MediapipeData, etc.).

   Warning: objects should never be initialized with this class directly,
   only when calling super().__init__() from the subclass (like `FlameData`).

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

   .. py:method:: mats2params(self, to_df=True)

      Transforms a time series (of length T) 4x4 affine matrices to a
      pandas DataFrame with a time series of T x 12 affine parameters
      (translation XYZ, rotation XYZ, scale XYZ, shear XYZ).


   .. py:method:: params2mats(self, params)

      Does the oppose as the above function.


   .. py:method:: save(self, path)

      Saves data to disk as a hdf5 file.

      :param path: Path to save the data to
      :type path: str


   .. py:method:: load(path)
      :staticmethod:

      Loads a hdf5 file from disk and returns a Data object.


   .. py:method:: events_to_mne(self)

      Converts events DataFrame to (N x 3) array that
      MNE expects.

      :returns: **events** -- An N (number of trials) x 3 array, with the first column
                indicating the sample *number* indicating the
      :rtype: np.ndarray


   .. py:method:: to_mne_rawarray(self)

      Creates an MNE `RawArray` object from the vertices (`v`).


   .. py:method:: render_video(self, f_out, renderer, video=None, scaling=None, n_frames=None, alpha=None)

      Should be implemented in subclass!


   .. py:method:: plot_data(self, f_out, plot_motion=True, plot_pca=True, n_pca=3)

      Creates a plot of the motion (rotation & translation) parameters
      over time and the first `n_pca` PCA components of the
      reconstructed vertices. For FLAME estimates, these parameters are
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


   .. py:method:: __len__(self)


   .. py:method:: __getitem__(self, idx)


   .. py:method:: __setitem__(self, idx, v)



.. py:class:: FlameData(*args, **kwargs)

   Bases: :py:obj:`BaseData`

   Base Data class with attributes and methods common to all Data
   classes (such as FlameData, MediapipeData, etc.).

   Warning: objects should never be initialized with this class directly,
   only when calling super().__init__() from the subclass (like `FlameData`).

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

   .. py:method:: load(cls, path)
      :classmethod:

      Loads a hdf5 file from disk and returns a Data object.


   .. py:method:: render_video(self, f_out, smooth=False, wireframe=False, **kwargs)

      Should be implemented in subclass!



.. py:class:: MediapipeData(*args, **kwargs)

   Bases: :py:obj:`BaseData`

   Base Data class with attributes and methods common to all Data
   classes (such as FlameData, MediapipeData, etc.).

   Warning: objects should never be initialized with this class directly,
   only when calling super().__init__() from the subclass (like `FlameData`).

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

   .. py:method:: load(cls, path)
      :classmethod:

      Loads a hdf5 file from disk and returns a Data object.


   .. py:method:: render_video(self, f_out, smooth=False, wireframe=False, **kwargs)

      Should be implemented in subclass!



.. py:class:: FANData(v, f=None, mat=None, cam_mat=None, frame_t=None, events=None, sf=None, img_size=None, recon_model_name=None, space='world', path=None)

   Bases: :py:obj:`BaseData`

   Base Data class with attributes and methods common to all Data
   classes (such as FlameData, MediapipeData, etc.).

   Warning: objects should never be initialized with this class directly,
   only when calling super().__init__() from the subclass (like `FlameData`).

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

   .. py:method:: load(cls, path)
      :classmethod:

      Loads a hdf5 file from disk and returns a Data object.


   .. py:method:: render_video(self, f_out, video=None, margin=25)

      Should be implemented in subclass!



.. py:data:: MODEL2CLS
   

   

.. py:function:: load_h5(path)

   Convenience function to load a hdf5 file and
   immediately initialize the correct data class.

   Located here (instead of io.py or render.py) to
   prevent circular imports.

   :param path: Path to hdf5 file
   :type path: str

   :returns: **data** -- An object with a class derived from data.BaseData
             (like MediapipeData, or FlameData)
   :rtype: data.BaseData subclass object


