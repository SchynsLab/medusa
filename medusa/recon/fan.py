import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from matplotlib.patches import Rectangle
from face_alignment import LandmarksType, FaceAlignment
from skimage.transform import estimate_transform, warp, resize

from ..utils import get_logger

logger = get_logger()


class FAN:
    """FAN face detection and landmark estimation, as implemented by
    Bulat & Tzimiropoulos (2017, Arxiv), adapted for use with DECA
    by Yao Feng (https://github.com/YadiraF), and further modified by
    Lukas Snoek

    Parameters
    ----------
    device : str
        Device to use, either 'cpu' or 'cuda' (for GPU)
    target_size : int
        Size to crop the image to, assuming a square crop; default (224)
        corresponds to image size that DECA expects
    face_detector : str
        Face detector algorithm to use (default: 'sfd', as used in DECA)
    lm_type : str
        Either '2D' (using 2D landmarks) or '3D' (using 3D landmarks)
    use_prev_fan_bbox : bool
        Whether to use the previous bbox from FAN to do an initial crop (True)
        or whether to run the FAN face detection algorithm again (False)
    use_prev_bbox : bool
        Whether to use the previous DECA-style bbox (True) or whether to
        run FAN again to estimate landmarks from which to create a new
        bbox (False); this should only be used when there is very little
        rigid motion of the face!
    """

    def __init__(
        self,
        device="cpu",
        target_size=224,
        face_detector="sfd",
        lm_type="2D",
        use_prev_fan_bbox=False,
        use_prev_bbox=False,
    ):

        self.model = FaceAlignment(
            getattr(LandmarksType, f"_{lm_type}"),
            flip_input=False,
            device=device,
            face_detector=face_detector,
        )
        self.device = device
        self.target_size = target_size
        self.img_orig = None
        self.use_prev_fan_bbox = use_prev_fan_bbox
        self.use_prev_bbox = use_prev_bbox
        self.prev_fan_bbox = None
        self.bbox = None
        self.cropped_image = None
        self.tform = None

    def _load_image(self, image):
        """Loads image using PIL if it's not already
        a numpy array."""
        if isinstance(image, (str, Path)):
            image = np.array(imread(image))

        return image

    def prepare_for_emoca(self, image):
        """Runs all steps of the cropping / preprocessing pipeline
        necessary for use with DECA/EMOCA."""

        self.img_orig = self._load_image(image)

        if self.bbox is None or not self.use_prev_bbox:
            # Only run this when we want a new bbox or
            # when there isn't a bbox estimated yet
            self.__call__()
            self._create_bbox()

        self._crop()
        img = self._preprocess()
        return img

    def __call__(self, image=None):
        """Estimates (2D) landmarks (vertices) on the face.

        Parameters
        ----------
        image : str, Path, or numpy array
            Path (str or pathlib Path) pointing to image file or 3D numpy array
            (with np.uint8 values) representing a RGB image

        Raises
        ------
        ValueError : if `image` is `None` *and* self.img_orig is `None`
        """

        if image is not None:
            self.img_orig = self._load_image(image)
        else:
            if self.img_orig is None:
                raise ValueError(
                    "If no `image` is supplied, `self.img_orig` should exist!"
                )

        # First try with (optionally) the previous FAN bbox
        prev_fan_bbox = self.prev_fan_bbox if self.use_prev_fan_bbox else None
        lm, _, fan_bbox = self.model.get_landmarks_from_image(
            self.img_orig, detected_faces=prev_fan_bbox, return_bboxes=True
        )

        if lm is None and self.prev_fan_bbox is not None:
            # Second try: without previous FAN bbox
            lm, _, fan_bbox = self.model.get_landmarks_from_image(
                self.img_orig, return_bboxes=True
            )

        if lm is None:
            # It still didn't work, raise Error (maybe just warning?)
            raise ValueError("No face detected!")
        elif len(lm) > 1:
            raise ValueError(f"More than one face (i.e., {len(lm)}) detected!")
        else:
            lm = lm[0]

        self.lm = lm  # used by _create_bbox
        self.prev_fan_bbox = fan_bbox
        return {"v": lm}

    def _create_bbox(self, scale=1.25):
        """Creates a bounding box (bbox) based on the landmarks by creating
        a box around the outermost landmarks (+10%), as done in the original
        DECA usage.

        Parameters
        ----------
        scale : float
            Factor to scale the bounding box with
        """
        left = np.min(self.lm[:, 0])
        right = np.max(self.lm[:, 0])
        top = np.min(self.lm[:, 1])
        bottom = np.max(self.lm[:, 1])

        orig_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(orig_size * scale)
        self.bbox = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],  # bottom left
                [center[0] - size / 2, center[1] + size / 2],  # top left
                [center[0] + size / 2, center[1] - size / 2],  # bottom right
                [center[0] + size / 2, center[1] + size / 2],
            ]
        )  # top right

    def _crop(self):
        """Using the bounding box (`self.bbox`), crops the image by warping the image
        based on a similarity transform of the bounding box to the corners of target size
        image."""
        w, h = self.target_size, self.target_size
        dst = np.array([[0, 0], [0, w - 1], [h - 1, 0]])
        self.tform = estimate_transform("similarity", self.bbox[:3, :], dst)

        # Note to self: preserve_range needs to be True, because otherwise `warp` will scale the data!
        self.img_crop = warp(
            self.img_orig, self.tform.inverse, output_shape=(w, h), preserve_range=True
        )

    def _preprocess(self):
        """Resizes, tranposes (channels, width, height), rescales (/255) the data,
        casts the data to torch, and add a batch dimension (`unsqueeze`)."""
        img = resize(
            self.img_crop, (self.target_size, self.target_size), anti_aliasing=True
        )
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        img = torch.tensor(img).float().to(self.device)
        return img.unsqueeze(0)  # add singleton batch dim

    def viz_qc(self, f_out=None, return_rgba=False):
        """Visualizes the inferred 2D landmarks & bounding box, as well as the final
        cropped image.

        f_out : str, Path
            Path to save viz to
        return_rgba : bool
            Whether to return a numpy image with the raw pixel RGBA intensities
            (True) or not (False; return nothing)
        """

        if f_out is None and return_rgba is False:
            raise ValueError("Either supply f_out or set return_rgb to True!")

        fig, axes = plt.subplots(nrows=2, constrained_layout=True)
        axes[0].imshow(self.img_orig)
        axes[0].axis("off")
        axes[0].plot(self.lm[:, 0], self.lm[:, 1], marker="o", ms=2, ls="")

        w = self.bbox[2, 0] - self.bbox[0, 0]
        h = self.bbox[3, 1] - self.bbox[2, 1]
        rect = Rectangle(self.bbox[0, :], w, h, facecolor="none", edgecolor="r")
        axes[0].add_patch(rect)

        axes[1].imshow(self.img_crop.astype(np.uint8))
        axes[1].axis("off")

        if f_out is not None:
            fig.savefig(f_out)
            plt.close()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format="raw", dpi=100)
            io_buf.seek(0)
            img = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
            )
            io_buf.close()
            plt.close()
            return img
