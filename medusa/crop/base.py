import cv2
import torch
import numpy as np
from pathlib import Path

from ..io import load_inputs


class BaseCropModel:

    def to_numpy(self, img):
        """ 'Undoes' the preprocessing of the cropped image and returns an ordinary
        h x w x 3 numpy array. Useful for checking the cropping result. 
        
        Parameters
        ----------
        img : torch.Tensor
            The result from the cropping operation (i.e., whatever the ``__call__``
            method returns); should be a 1 x 3 x 224 x 224 tensor
        
        Returns
        -------
        img : np.ndarray
            A 224 x 224 x 3 numpy array with uint8 values
        """        
        
        img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        img = ((img * self._scale) + self._mean).astype(np.uint8)

        if not self._to_bgr:
            img = img[:, :, :, ::-1]

        return img
    
    def close(self):
        
        if hasattr(self, '_warned_about_multiple_faces'):
            self._warned_about_multiple_faces = False

    def _load_inputs(self, inputs, *args, **kwargs):
        """ Utility function to load images from paths if the model is not provided
        with a [batch, w, h, 3] tensor. """
        return load_inputs(inputs, *args, **kwargs)

    @staticmethod
    def visualize_bbox(image, bbox, f_out):
        """ Creates an image with the estimated bounding box (bbox) on top of it.
        
        Parameters
        ----------
        image : np.ndarray
            A numpy array with the original (uncropped images); can also be
            a torch Tensor; can be a batch of images or a single image
        bbox : np.ndarray
            A numpy array with the bounding box(es) corresponding to the
            image(s)
        f_out : str, pathlib.Path
            If multiple images, a number (_xxx) is appended
        """
        if not isinstance(f_out, Path):
            f_out = Path(f_out)

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.ndim == 3:
            image = image[None, ...]

        if bbox.shape[-1] == 4:
            if bbox.ndim == 1:
                bbox = bbox[None, :]
        else:
            if bbox.ndim == 2:
                bbox = bbox[None, :, :]

        bbox = bbox.round().astype(int)
        for i in range(image.shape[0]):
            image_ = cv2.cvtColor(image[i, ...], cv2.COLOR_RGB2BGR)
            bbox_ = bbox[i, ...]

            if bbox_.size == 4:
                cv2.rectangle(image_, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(image_, bbox_[0, :], bbox_[3, :], (0, 0, 255), 2)

            if image.shape[0] == 1:
                this_f_out = f_out
            else:
                this_f_out = f_out.stem + f'_{i:02d}{f_out.suffix}'

            cv2.imwrite(str(this_f_out), image_)
