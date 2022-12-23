"""Module with a wrapper around a Mediapipe face mesh model [1]_ that can be
used in Medusa.

.. [1] Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M.
(2019).        Real-time facial surface geometry from monocular video on
mobile GPUs. *arXiv preprint arXiv:1907.06724*
"""

from collections import defaultdict

import numpy as np
import torch

from ...defaults import DEVICE
from ...data import get_template_mediapipe
from ...io import load_inputs
from ..base import BaseReconModel
from ._transforms import PCF, image2world


class Mediapipe(BaseReconModel):
    """A Mediapipe face mesh reconstruction model.

    Parameters
    ----------
    static_image_mode : bool
        Whether to expect a sequence of related images (like in a video)
    **kwargs : dict
        Extra keyword arguments to be passed to the initialization of FaceMesh

    Attributes
    ----------
    model : mediapipe.solutions.face_mesh.FaceMesh
        The actual Mediapipe model object
    """

    def __init__(
        self,
        static_image_mode=False,
        refine_landmarks=True,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.5,
        device=DEVICE,
    ):
        """Initializes a Mediapipe recon model."""

        # Importing here speeds up CLI
        import mediapipe as mp

        self.model = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_faces=5,
        )

        self.model.__enter__()  # enter context manually
        self.device = device
        self._pcf = None  # initialized later
        self._load_reference()  # sets self.{v,f}_world_ref

    def __str__(self):
        return "mediapipe"

    def _load_reference(self):
        """Loads the vertices and faces of the references template in world
        space."""

        out = get_template_mediapipe()
        self._v_world_ref = out["v"]
        self._f_world_ref = out["tris"]

    def get_tris(self):

        return torch.as_tensor(self._f_world_ref, device=self.device)

    def get_cam_mat(self):
        return torch.eye(4, device=self.device)

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

        imgs = load_inputs(
            imgs,
            load_as="numpy",
            channels_first=False,
            with_batch_dim=True,
            dtype="uint8",
        )

        outputs = defaultdict(list)
        for i in range(imgs.shape[0]):
            img = imgs[i, ...]
            img.flags.writeable = False
            results = self.model.process(img)
            if not results.multi_face_landmarks:
                continue

            for detection in results.multi_face_landmarks:
                lm = detection.landmark

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

                outputs["v"].append(v_[:, :3])
                outputs["mat"].append(mat_)
                outputs["img_idx"].append(i)

        outputs["n_img"] = imgs.shape[0]
        if outputs.get("v", None) is not None:
            outputs["v"] = np.stack(outputs["v"]).astype(np.float32)
            outputs["mat"] = np.stack(outputs["mat"]).astype(np.float32)
            outputs["img_idx"] = np.array(outputs["img_idx"])

            for attr, data in outputs.items():
                outputs[attr] = torch.as_tensor(data, device=self.device)

        return outputs

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
