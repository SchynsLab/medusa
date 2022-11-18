import torch
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from skimage.transform import estimate_transform, warp

from .base import BaseCropModel


class FanCropModel(BaseCropModel):
    """ Cropping model based on the estimated (2D) landmarks from  the ``face_alignment``
    package.
    
    Parameters
    ----------
    device : str
        Either 'cuda' (GPU) or 'cpu'
    target_size : tuple
        Length 2 tuple with desired width/heigth of cropped image; should be (224, 224)
        for EMOCA and DECA
    
    Attributes
    ----------
    model : FaceAlignment
        The initialized face alignment model from ``face_alignment``, using 2D landmarks    
    """

    def __init__(self, device='cuda', target_size=(224, 224), min_detection_confidence=0.5, return_bbox=False):
        self.device = device
        self.target_size = target_size
        self.return_bbox = return_bbox
        self.model = FaceAlignment(LandmarksType._2D, device=device,
                                   face_detector_kwargs={'filter_threshold': min_detection_confidence})
        self._warned_about_multiple_faces = False
        self._mean = 0.
        self._scale = 255.
        self._to_bgr = False

    def _create_bbox(self, lm, scale=1.25):
        """ Creates a bounding box (bbox) based on the landmarks by creating
        a box around the outermost landmarks (+10%), as done in the original
        DECA usage.

        Parameters
        ----------
        scale : float
            Factor to scale the bounding box with
        """
        left = np.min(lm[:, 0])
        right = np.max(lm[:, 0])
        top = np.min(lm[:, 1])
        bottom = np.max(lm[:, 1])

        orig_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(orig_size * scale)
        return np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],  # bottom left
                [center[0] - size / 2, center[1] + size / 2],  # top left
                [center[0] + size / 2, center[1] - size / 2],  # bottom right
                [center[0] + size / 2, center[1] + size / 2],  # top right
            ]
        )  

    def _get_area(self, bbox):
        """ Computes the area of a bounding box in pixels. """         
        nx = bbox[2, 0] - bbox[0, 0]
        ny = bbox[1, 1] - bbox[0, 1]
        
        return nx * ny

    def _crop(self, img_orig, bbox):
        """ Using the bounding box (`self.bbox`), crops the image by warping the image
        based on a similarity transform of the bounding box to the corners of target size
        image. """
        w, h = self.target_size
        dst = np.array([[0, 0], [0, w - 1], [h - 1, 0]])
        crop_mat = estimate_transform("similarity", bbox[:3, :], dst)

        # Note to self: preserve_range needs to be True, because otherwise `warp` will scale the data!
        img_crop = warp(img_orig, crop_mat.inverse, output_shape=(w, h), preserve_range=True)
        return img_crop, crop_mat

    def _preprocess(self, img_crop, crop_mat):
        """ Transposes (channels, width, height), rescales (/255) the data,
        casts the data to torch, and add a batch dimension (`unsqueeze`). """
        
        img_crop = img_crop.transpose(0, 3, 1, 2)
        img_crop = img_crop / self._scale
        img_crop = torch.tensor(img_crop, dtype=torch.float32).to(self.device)
        crop_mat = torch.tensor(crop_mat, dtype=torch.float32).to(self.device)
        return img_crop, crop_mat

    def __call__(self, images):
        """ Runs all steps of the cropping / preprocessing pipeline
        necessary for use with Flame-based models such as DECA/EMOCA. 
        
        Parameters
        -----------
        images : str, Path, list, torch.Tensor
            A single path or list of paths to one or more images, or 
            already loaded images as a ``torch.Tensor`` (as yielded by
            a ``VideoLoader`` object)
        
        Returns
        -------
        img_crop : torch.Tensor
            The preprocessed (normalized) and cropped image as a ``torch.Tensor``
            of shape (batch_dim, 3, 224, 224), as DECA-based models expect
        crop_mat : torch.Tensor
            The cropping matres, a (batch_dim, 3, 3) tensor with affine matrices
        
        Examples
        --------
        To preprocess (which includes cropping) an image and get the cropping matrix:
        
        >>> from medusa.data import get_example_frame
        >>> crop_model = FanCropModel(device='cpu')
        >>> img = get_example_frame()  # path to jpg image
        >>> cropped_img, crop_mat = crop_model(img)
        >>> cropped_img.shape
        torch.Size([1, 3, 224, 224])
        >>> crop_mat.shape
        torch.Size([1, 3, 3])

        To preprocess an set of video frames using the ``VideoLoader`` object:

        >>> from medusa.data import get_example_video
        >>> crop_model = FanCropModel(device='cpu')
        >>> vid_loader = get_example_video(return_videoloader=True, batch_size=16)
        >>> img_batch = next(vid_loader)
        >>> img_crop, crop_mat = crop_model(img_batch)
        >>> img_crop.shape
        torch.Size([16, 3, 224, 224])
        >>> crop_mat.shape
        torch.Size([16, 3, 3])
        """

        images = self._load_inputs(images, to_bgr=self._to_bgr, channels_first=True, device=self.device)

        if images.ndim == 3:
            # Need a batch dim!
            images = images.unsqueeze(0)

        if images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2)

        # Estimate landmarks
        lms = self.model.get_landmarks_from_batch(images)
        img_crop = np.zeros((len(lms), *self.target_size, 3))
        crop_mat = np.zeros((len(lms), 3, 3))
        bbox = np.zeros((len(lms), 4, 2))

        for i, lm in enumerate(lms):
            # if len(lm) > 1:
            #     if not self._warned_about_multiple_faces:
            #         logger.warning(f"More than one face (i.e., {len(lm)}) detected; "
            #                     "picking largest one!")
            #         self._warned_about_multiple_faces = True

            #     # Definitely not foolproof, but pick the face with the biggest 
            #     # bounding box (alternative idea: correlate with canonical bbox)
            #     bbox = [self._create_bbox(lm_) for lm_ in lm]
            #     areas = np.array([self._get_area(bb) for bb in bbox])
            #     idx = areas.argmax()
            #     lm, bbox = lm[idx], bbox[idx]                        
            # else:
            #    lm = lm[0]
            bbox[i, ...] = self._create_bbox(lm)
            
            # Create bounding box based on landmarks, use that to crop image, and return
            # preprocessed (normalized, to tensor) image
            img_orig = images[i, ...].cpu().numpy().transpose(1, 2, 0)
            img_crop[i, ...], crop_mat[i, ...] = self._crop(img_orig, bbox[i])

        img_crop, crop_mat = self._preprocess(img_crop, crop_mat)
        if self.return_bbox:
            return img_crop, crop_mat, bbox
        else:
            return img_crop, crop_mat