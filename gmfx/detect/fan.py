import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from matplotlib.patches import Rectangle
from face_alignment import LandmarksType, FaceAlignment
from skimage.transform import estimate_transform, warp, resize


class FAN(object):
    """ FAN face detection and landmark estimation, as implemented by
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
    """
    def __init__(self, device='cpu', target_size=224, face_detector='sfd'):
        self.model = FaceAlignment(LandmarksType._2D, flip_input=False, device=device,
                                   face_detector=face_detector)
        self.device = device
        self.target_size = target_size
        self.image = None
        self.cropped_image = None
        self.tform = None

    def __call__(self, image):
        """ Runs all steps of the cropping / preprocessing pipeline
        necessary for use with DECA. """
        self.get_landmarks(image)
        self.create_bbox()
        self.crop()
        img = self.preprocess()
        return img

    def get_landmarks(self, image):
        """ Estimates (2D) landmarks on the face.
        
        Parameters
        ----------
        image : str, Path, or numpy array
            Path (str or pathlib Path) pointing to image file or 3D numpy array
            (with np.uint8 values) representing a RGB image
        """
        
        if isinstance(image, (str, Path)):
            image = np.array(imread(image))
        
        self.image = image  # store for later use
        lm2d = self.model.get_landmarks_from_image(self.image)
        
        if lm2d is None:
            raise ValueError("No face detected!")
        elif len(lm2d) > 1:
            raise ValueError(f"More than one face (i.e., {len(lm2d)}) detected!")
        else:
            lm2d = lm2d[0]
        
        self.lm2d = lm2d
        
    def create_bbox(self, scale=1.25):
        """ Creates a bounding box (bbox) based on the landmarks by creating
        a box around the outermost landmarks (+10%). 
        
        Parameters
        ----------
        scale : float
            Factor to scale the bounding box with
        """        
        left = np.min(self.lm2d[:, 0])
        right = np.max(self.lm2d[:, 0]) 
        top = np.min(self.lm2d[:, 1])
        bottom = np.max(self.lm2d[:, 1])
        
        orig_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = int(orig_size * scale)
        self.bbox = np.array([[center[0]-size/2, center[1]-size/2],    # bottom left
                              [center[0]-size/2, center[1]+size/2],    # top left
                              [center[0]+size/2, center[1]-size/2],    # bottom right
                              [center[0]+size/2, center[1]+size/2]])   # top right
    
    def crop(self):
        """ Using the bounding box (`self.bbox`), crops the image by warping the image
        based on a similarity transform of the bounding box to the corners of target size
        image. """
        w, h = self.target_size, self.target_size
        dst = np.array([[0, 0], [0, w - 1], [h - 1, 0]])
        self.tform = estimate_transform('similarity', self.bbox[:3, :], dst)
        
        # Cast tform_params to torch to be used in DECA
        self.tform_params = torch.tensor(self.tform.params).float()[None, ...]
        self.tform_params = torch.inverse(self.tform_params).transpose(1, 2).to(self.device)
        
        # Note to self: preserve_range needs to be True, because otherwise `warp` will scale the data!
        self.cropped_image = warp(self.image, self.tform.inverse, output_shape=(w, h), preserve_range=True)

    def preprocess(self):
        """ Resizes, tranposes (channels, width, height), rescales (/255) the data,
        casts the data to torch, and add a batch dimension (`unsqueeze`). """
        img = resize(self.cropped_image, (self.target_size, self.target_size), anti_aliasing=True)        
        img = img.transpose(2, 0, 1)
        img = img / 255.    
        img = torch.tensor(img).float().to(self.device)
        return img.unsqueeze(0)
    
    def viz(self, f_out):
        """ Visualizes the inferred 2D landmarks & bounding box, as well as the final
        cropped image."""

        fig, axes = plt.subplots(ncols=2, constrained_layout=True)
        axes[0].imshow(self.image)
        axes[0].axis('off')
        axes[0].plot(self.lm2d[:, 0], self.lm2d[:, 1], marker='o', ms=2, ls='')

        w = self.bbox[2, 0] - self.bbox[0, 0]
        h = self.bbox[3, 1] - self.bbox[2, 1]
        rect = Rectangle(self.bbox[0, :], w, h, facecolor='none', edgecolor='r')
        axes[0].add_patch(rect)
        
        axes[1].imshow(self.cropped_image)
        axes[1].axis('off')
        fig.savefig(f_out)
        plt.close()