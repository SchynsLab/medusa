""" Module with a wrapper around a Mediapipe face mesh model [1]_ that can be
used in Medusa. 

.. [1] Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). 
   Real-time facial surface geometry from monocular video on mobile GPUs. arXiv
   preprint arXiv:1907.06724
""" 

import numpy as np
from pathlib import Path
from trimesh.exchange.obj import load_obj

from .transforms import PCF, image2world


class Mediapipe:
    """ A Mediapipe face mesh reconstruction model.

    Parameters
    ----------
    static_image_mode : bool
        Whether to expect a sequence of related images
        (like in a video)
    kwargs : dict
        Extra keyword arguments to be passed to
        the initialization of FaceMesh
    
    Attributes
    ----------
    model : mediapipe.solutions.face_mesh.FaceMesh
        The actual Mediapipe model object

    """ 
    
    def __init__(self, static_image_mode=True, **kwargs):
        """ Initializes a Mediapipe recon model. """

        # Importing here speeds up CLI
        import mediapipe as mp

        self.model = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            **kwargs
        )
        self.model.__enter__()  # enter context manually
        self._pcf = None  # initialized later
        self._load_reference()  # sets self.{v,f}_world_ref

    def _load_reference(self):
        """ Loads the vertices and faces of the references template
        in world space. """

        path = Path(__file__).parents[2] / "data/mediapipe_template.obj"
        with open(path, "r") as f_in:
            obj = load_obj(f_in)
            self._v_world_ref = obj["vertices"]
            self._f_world_ref = obj["faces"]

    def __call__(self, image):
        """ Performs reconstruction of the face as a list of landmarks (vertices).

        Parameters
        ----------
        image : np.ndarray
            A 3D (w x h x 3) numpy array representing a RGB image

        Returns
        -------
        out : dict
            A dictionary with two keys: ``"v"``, the reconstructed vertices (468 in 
            total) and ``"mat"``, a 4x4 Numpy array representing the local-to-world
            matrix
        
        Notes
        -----
        This implementation returns 468 vertices instead of the original 478, because
        the last 10 vertices (representing the irises) are not present in the canonical
        model.       
        
        Examples
        --------
        To reconstruct an example, simply call the ``Mediapipe`` object:
        
        >>> from medusa.data import get_example_frame
        >>> model = Mediapipe()
        >>> img = get_example_frame()
        >>> out = model(img)  # reconstruct!
        >>> out['v'].shape    # vertices
        (468, 3)
        >>> out['mat'].shape  # local-to-world matrix
        (4, 4)
        """
        results = self.model.process(image)

        if not results.multi_face_landmarks:
            raise ValueError("Could not reconstruct face!")
        elif len(results.multi_face_landmarks) > 1:
            raise ValueError("Found more than 1 face!")
        else:
            lm = results.multi_face_landmarks[0].landmark

        # Extract coordinates of all landmarks
        x = np.array([lm_.x for lm_ in lm])
        y = np.array([lm_.y for lm_ in lm])
        z = np.array([lm_.z for lm_ in lm])
        v = np.c_[x, y, z]  # 478 (landmarks x 3 (x, y, z)

        if self._pcf is None:
            # Because we need the image dimensions, we need to initialize the
            # (pseudo)camera here (but we assume image dims are constant for video)
            self._pcf = PCF(
                frame_height=image.shape[0],
                frame_width=image.shape[1],
                fy=image.shape[1],
            )

        # Canonical (reference) model does not have iris landmarks (last ten),
        # so remove these before inputting it into function
        v = v[:468, :]

        # Project vertices back into world space using a Python implementation by
        # Rasmus Jones (https://github.com/Rassibassi/mediapipeDemos/blob/main/head_posture.py)
        v, mat = image2world(v.T.copy(), self._pcf, self._v_world_ref.T)

        # Add back translation and rotation to the vertices
        v = np.c_[v.T, np.ones(468)] @ mat.T
        v = v[:, :3]

        out = {'v': v, 'mat': mat}
        return out

        # For posterity, if you want to render v_world into pixel space using `pyrender`,
        # use the IntrinsicsCamera object with parameters:
        # fx=img.shape[1], fy=img.shape[1], cx=img.shape[1] / 2, cy=img.shape[0] / 2
        # Mediapipe assumes that the camera is located at the origin (and pointing in -z),
        # so no need to set the camera pose matrix (extrinsic camera matrix)

    def close(self):
        """ Closes context manager.
        
        Ideally, after you're doing with reconstructing each frame of the video,
        you call this method to close the manually opened context (but shouldn't
        matter much if you only instantiate a single model).       
        """
        # Note: __exit__ just calls close()
        self.model.__exit__(None, None, None)
