# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import numpy as np
import face_alignment
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from matplotlib.patches import Rectangle
from skimage.transform import estimate_transform, warp, resize


class FAN(object):
    def __init__(self, device='cpu', target_size=224):
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device, face_detector='sfd')
        self.device = device
        self.target_size = target_size
        self.image = None
        self.cropped_image = None
        self.tform = None

    def __call__(self, image):
        self.get_landmarks(image)
        self.create_bbox()
        self.crop()
        img = self.preprocess()
        return img

    def get_landmarks(self, image):
        
        if isinstance(image, (str, Path)):
            image = np.array(imread(image))
        
        self.image = image
        lm2d = self.model.get_landmarks_from_image(self.image)
        
        if lm2d is None:
            raise ValueError("No face detected!")
        elif len(lm2d) > 1:
            #lm2d = lm2d[0]
            raise ValueError(f"More than one face (i.e., {len(lm2d)}) detected!")
        else:
            lm2d = lm2d[0]
        
        self.lm2d = lm2d
        
    def create_bbox(self, scale=1.25):
        
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
        image."""
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
        casts the data to torch, and add a batch dimension (`unsqueeze`)."""
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
    