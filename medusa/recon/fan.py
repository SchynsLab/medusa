""" Module with functionality to use the FAN-3D model.

This module contains a reconstruction model based on the ``face_alignment`` package
by `Adrian Bulat <https://www.adrianbulat.com/face-alignment>`_ [1]_. It is used both
as a reconstruction model as well as a way to estimate a bounding box as expected by
the EMOCA model (which uses the bounding box to crop the original image).

.. [1] Bulat, A., & Tzimiropoulos, G. (2017). How far are we from solving the 2d & 3d
       face alignment problem?(and a dataset of 230,000 3d facial landmarks).
       In *Proceedings of the IEEE International Conference on Computer Vision*
       (pp. 1021-1030).
"""

import numpy as np
from pathlib import Path
from skimage.io import imread

from ..utils import get_logger
from .base import BaseModel

logger = get_logger()


class FAN(BaseModel):
    """ A wrapper around the FAN-3D landmark prediction model.
    
    Parameters
    ----------
    device : str
        Device to use, either 'cpu' or 'cuda' (for GPU)
    use_prev_bbox : bool
        Whether to use the previous bbox from FAN to do an initial crop (True)
        or whether to run the FAN face detection algorithm again (False)
    
    Attributes
    ----------
    model : face_alignment.FaceAlignment
        The actual face alignment model

    Examples
    --------
    To create a FAN based reconstruction model:
    
    >>> recon_model = FAN(device='cpu')
    
    """
    
    def __init__(self, device="cpu", use_prev_bbox=True, min_detection_threshold=0.5,
                 **kwargs):
        """ Initializes a FAN object. """

        # Import face_alignment here instead of on top of module to avoid loading
        # torch (takes long) when we don't need to
        from face_alignment import LandmarksType, FaceAlignment
        
        face_det_kwargs = {'filter_threshold': min_detection_threshold}
        self.model = FaceAlignment(LandmarksType._3D, device=device,
                                   face_detector_kwargs=face_det_kwargs, **kwargs)
        self.device = device
        self.use_prev_bbox = use_prev_bbox
        self.prev_bbox = None  # to store previous bounding box

    def _load_image(self, image):
        """Loads image using PIL if it's not already
        a numpy array."""
        if isinstance(image, (str, Path)):
            image = np.array(imread(image))

        return image

    def get_faces(self):
        """ FAN only returns landmarks, not a full mesh. """
        return None

    def __call__(self, image=None):
        """Estimates landmarks (vertices) on the face.

        Parameters
        ----------
        image : str, Path, np.ndarray
            Either a string or ``pathlib.Path`` object to an image or a numpy array
            (width x height x 3) representing the already loaded RGB image

        Returns
        -------
        out : dict
            A dictionary with one key: ``"v"``, the reconstructed vertices (68 in 
            total) with 2 (if using ``lm_type='2D'``) or 3 (if using ``lm_type='3D'``)
            coordinates

        Examples
        --------
        To reconstruct an example, simply call the ``FAN`` object:
        
        >>> from medusa.data import get_example_frame
        >>> model = FAN(device='cpu')
        >>> img = get_example_frame()
        >>> out = model(img)  # reconstruct!
        >>> out['v'].shape    # vertices
        (68, 3)
        """

        img = self._load_image(image)

        # First try with (optionally) the previous FAN bbox
        prev_bbox = self.prev_bbox if self.use_prev_bbox else None
        lm, _, bbox = self.model.get_landmarks_from_image(
            img.copy(), detected_faces=prev_bbox, return_bboxes=True
        )

        if lm is None and self.prev_bbox is not None:
            # Second try: without previous FAN bbox
            lm, _, bbox = self.model.get_landmarks_from_image(
                img.copy(), return_bboxes=True
            )

        if lm is None:
            # It still didn't work, raise Error (maybe just warning?)
            raise ValueError("No face detected!")
        elif len(lm) > 1:
            raise ValueError(f"More than one face (i.e., {len(lm)}) detected!")
        else:
            lm = lm[0]

        self.lm = lm  # used by _create_bbox
        self.prev_bbox = bbox
        return {"v": lm}
