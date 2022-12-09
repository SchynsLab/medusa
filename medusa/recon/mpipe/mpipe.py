"""Module with a wrapper around a Mediapipe face mesh model [1]_ that can be
used in Medusa.

.. [1] Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M.
(2019).    Real-time facial surface geometry from monocular video on
mobile GPUs. *arXiv    preprint arXiv:1907.06724*
"""

import numpy as np

from ...data import get_template_mediapipe
from ..base import BaseReconModel
from ._transforms import PCF, image2world


class Mediapipe(BaseReconModel):
    """A Mediapipe face mesh reconstruction model.

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

    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.01, min_tracking_confidence=0.1):
        """Initializes a Mediapipe recon model."""

        # Importing here speeds up CLI
        import mediapipe as mp
        self.model = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.model.__enter__()  # enter context manually
        self._pcf = None  # initialized later
        self._load_reference()  # sets self.{v,f}_world_ref

    def _load_reference(self):
        """Loads the vertices and faces of the references template in world
        space."""

        out = get_template_mediapipe()
        self._v_world_ref = out['v']
        self._f_world_ref = out['f']

    def get_tris(self):
        return self._f_world_ref

    def __call__(self, imgs):
        """Performs reconstruction of the face as a list of landmarks
        (vertices).

        Parameters
        ----------
        imgs : np.ndarray
            A 4D (b x w x h x 3) numpy array representing a batch of RGB images

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
        (1, 468, 3)
        >>> out['mat'].shape  # local-to-world matrix
        (1, 4, 4)
        """

        imgs = self._load_inputs(imgs, load_as='numpy', channels_first=False,
                                 with_batch_dim=True, dtype='uint8')

        v = np.zeros((imgs.shape[0], 468, 3))
        mat = np.zeros((imgs.shape[0], 4, 4))
        for i in range(imgs.shape[0]):

            results = self.model.process(imgs[i, ...])
            if not results.multi_face_landmarks:
                return None
                #raise ValueError("Could not reconstruct face! Try decreasing `min_detection_confidence`")
            elif len(results.multi_face_landmarks) > 1:
                raise ValueError("Found more than 1 face!")
            else:
                lm = results.multi_face_landmarks[0].landmark

            # Extract coordinates of all landmarks
            x = np.array([lm_.x for lm_ in lm])
            y = np.array([lm_.y for lm_ in lm])
            z = np.array([lm_.z for lm_ in lm])
            v_ = np.c_[x, y, z]  # 478 (landmarks x 3 (x, y, z)

            if self._pcf is None:
                # Because we need the image dimensions, we need to initialize the
                # (pseudo)camera here (but we assume image dims are constant for video)
                self._pcf = PCF(
                    frame_height=imgs.shape[1],
                    frame_width=imgs.shape[2],
                    fy=imgs.shape[2],
                )

            # Canonical (reference) model does not have iris landmarks (last ten),
            # so remove these before inputting it into function
            v_ = v_[:468, :]

            # Project vertices back into world space using a Python implementation by
            # Rasmus Jones (https://github.com/Rassibassi/mediapipeDemos/blob/main/head_posture.py)
            v_, mat_ = image2world(v_.T.copy(), self._pcf, self._v_world_ref.T)

            # Add back translation and rotation to the vertices
            v_ = np.c_[v_.T, np.ones(468)] @ mat_.T

            v[i, ...] = v_[:, :3]
            mat[i, ...] = mat_

        out = {'v': v, 'mat': mat}
        return out

        # For posterity, if you want to render v_world into pixel space using `pyrender`,
        # use the IntrinsicsCamera object with parameters:
        # fx=img.shape[1], fy=img.shape[1], cx=img.shape[1] / 2, cy=img.shape[0] / 2
        # Mediapipe assumes that the camera is located at the origin (and pointing in -z),
        # so no need to set the camera matrix (extrinsic camera matrix)

    def close(self):
        """Closes context manager.

        Ideally, after you're doing with reconstructing each frame of
        the video, you call this method to close the manually opened
        context (but shouldn't matter much if you only instantiate a
        single model).
        """
        # Note: __exit__ just calls close()
        self.model.__exit__(None, None, None)
