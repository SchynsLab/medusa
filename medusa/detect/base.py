import torch
import PIL
import numpy as np
from pathlib import Path
from collections import defaultdict
from torchvision.ops import box_area
from torchvision.utils import draw_bounding_boxes, draw_keypoints, save_image
from torchvision.io import write_video

from .. import DEVICE, FONT
from ..io import load_inputs


import random
_colors = list(PIL.ImageColor.colormap.values())
random.shuffle(_colors)
                

class BaseDetectionModel:

    pass


class DetectionResults:

    def __init__(self, n_img, conf=None, bbox=None, lms=None, img_idx=None, device=DEVICE):
        """ Note to self: we need to know `n_img` (number of original images), because
        otherwise we cannot make back from the 'stacked' format to the unstacked one. """
        self.n_img = n_img
        self.conf = conf
        self.bbox = bbox
        self.lms = lms
        self.img_idx = img_idx
        self.device = device
        self.face_idx = None
        self._concat()

    def _concat(self):

        for attr in ['conf', 'bbox', 'lms', 'img_idx']:
            
            value = getattr(self, attr)
            if value is None:
                continue

            if isinstance(value, list):
                if len(value) == 0:
                    setattr(self, attr, None)
                else:
                    value = np.concatenate(value)
                    value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
                    setattr(self, attr, value)

                    if attr == 'img_idx':
                        self.img_idx = self.img_idx.long()

    def __len__(self):
        if self.conf is None:
            return 0
        else:
            return len(self.conf)

    @classmethod
    def from_batches(cls, batches, sort=True):
        
        conf = torch.concatenate([b.conf for b in batches if b.conf is not None])
        bbox = torch.concatenate([b.bbox for b in batches if b.bbox is not None])
        lms = torch.concatenate([b.lms for b in batches if b.lms is not None])

        n_img = batches[0].n_img
        for i in range(1, len(batches)):
            if batches[i].img_idx is not None:
                batches[i].img_idx += n_img
            n_img += batches[i].n_img

        img_idx = torch.concatenate([b.img_idx for b in batches if b.img_idx is not None])
        device = batches[-1].device
        out_det = cls(n_img, conf, bbox, lms, img_idx, device)

        if sort:
            out_det.face_idx = out_det.sort()
        
        return out_det

    def sort(self):
        
        face_idx = torch.zeros_like(self.img_idx, device=self.device).long()
        
        for i, i_img in enumerate(self.img_idx.unique()):
            # lms = all detected landmarks for this image
            det_idx = i_img == self.img_idx
            lms = self.lms[det_idx]
            n_det = lms.shape[0]
            # n_det x n_lms x 2 -> n_det x (n_lms x 2)
            lms = lms.reshape((n_det, -1))
            
            if i == 0:
                # First detection, initialize tracker
                tracker = lms
            else:
                # Compute the distance between each detection (lms) and currently tracked faces (tracker)
                dists = torch.cdist(lms.reshape((n_det, -1)), tracker)  # n_det x current_faces

                # Get minimum distance (min_dists) and corresponding index (min_idx)
                # for each detection
                min_dists, min_idx = dists.min(dim=1)  # n_det x 1

                # new_idx will represent which faces in this frame are probably 'new'
                # (not seen before)
                new_idx = torch.zeros_like(min_dists, dtype=torch.bool)

                # First, check whether multiple detections are assigned the
                # same tracker index (min_idx), which by definition cannot be true
                for min_ in min_idx.unique():
                    doubles = min_ == min_idx
                    if doubles.sum() == 1:
                        # If there's only only 'double', there's no problem
                        # so just continue
                        continue
                    
                    # Otherwise, tell 'new_idx' that the largest distance
                    # of the doubles is probably a new face
                    doubles = torch.where(doubles, min_dists, 0)
                    new_idx[doubles.argmax()] = True 
                
                # Regardless of 'doubles', set detections with an unreasonably
                # large distance to any of the tracked faces also to true
                # (i.e., being a new face)
                new_idx[min_dists > 100] = True  # n_det x 1
                
                # Update the tracked faces with the non-new detected faces
                tracker[min_idx[~new_idx]] = lms[~new_idx]

                # If there are new faces (> 0), add them to the tracker
                n_new = new_idx.sum()
                if n_new > 0:
                    # Update the min_index for the new faces with a new integer
                    min_idx[new_idx] = tracker.shape[0] + torch.arange(n_new, device=self.device)
                    # Add new faces to the tracker
                    tracker = torch.concatenate((tracker, lms[new_idx]))

                face_idx[det_idx] = min_idx
        
        return face_idx

    def visualize(self, imgs, f_out, video=False, **kwargs):
        """ Creates an image with the estimated bounding box (bbox) on top of it.
        
        Parameters
        ----------
        image : array_like
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

        # batch_size x h x w x 3
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=self.device, dtype='uint8')
        n_images = imgs.shape[0]

        for i_img in range(n_images):

            img = imgs[i_img, ...]
            if self.img_idx is None:
                continue
            elif i_img not in self.img_idx:
                continue

            idx = self.img_idx == i_img
            bbox = self.bbox[idx]
            font_size = int((box_area(bbox).min().sqrt() / 8).item())
            labels = [str(lab) for lab in self.conf[idx].cpu().numpy().round(3)]

            if self.face_idx is not None:
                labels = [f"{lab}, id = {self.face_idx[idx][i].item()}" for i, lab in enumerate(labels)]
                colors = [_colors[i_] for i_ in self.face_idx[idx].cpu().numpy().tolist()]
            else:
                colors = (255, 0, 0)

            img = draw_bounding_boxes(img, bbox, labels, colors, width=2, font=FONT, font_size=font_size)

            lms = self.lms[idx]            
            img = draw_keypoints(img, lms, colors=(0, 255, 0), radius=2)
            imgs[i_img] = img
        
        if f_out.is_file():
            # Remove if exists already
            f_out.unlink()

        if video:
            write_video(str(f_out), imgs.permute(0, 2, 3, 1).cpu(), fps=24)            
        else:
            save_image(imgs.float(), fp=f_out, normalize=True, **kwargs)
