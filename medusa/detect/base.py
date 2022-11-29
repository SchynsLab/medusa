import torch
import PIL
import numpy as np
from pathlib import Path
from matplotlib import cm
from torchvision.ops import box_area
from torchvision.utils import draw_bounding_boxes, draw_keypoints, save_image
from torchvision.io import write_video

from .. import DEVICE, FONT
from ..io import load_inputs
                

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
    def from_batches(cls, batches):
        
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
        return cls(n_img, conf, bbox, lms, img_idx, device)

    def sort(self, dist_threshold=200, present_threshold=0.1):
        
        face_idx = torch.zeros_like(self.img_idx, device=self.device, dtype=torch.int64)
        
        for i, i_img in enumerate(self.img_idx.unique()):
            # lms = all detected landmarks for this image
            det_idx = i_img == self.img_idx
            lms = self.lms[det_idx]

            # flatten landmarks (5 x 2 -> 10)
            n_det = lms.shape[0]
            lms = lms.reshape((n_det, -1))

            if i == 0:
                # First detection, initialize tracker with first landmarks
                face_idx[det_idx] = torch.arange(0, n_det, device=self.device)
                tracker = lms
                continue
            
            # Compute the distance between each detection (lms) and currently tracked faces (tracker)
            dists = torch.cdist(lms, tracker)  # n_det x current_faces

            # face_assigned keeps track of which detection is assigned to which face from
            # the tracker (-1 refers to "not assigned yet")
            face_assigned = torch.ones(n_det, device=self.device, dtype=torch.int64) * -1

            # We'll keep track of which tracked faces we have not yet assigned
            track_list = torch.arange(dists.shape[1], device=self.device)

            # Check the order of minimum distances across detections
            # (which we'll use to loop over)               
            order = dists.min(dim=1)[0].argsort()
            det_list = torch.arange(dists.shape[0], device=self.device)

            # Loop over detections, sorted from best match (lowest dist)
            # to any face in the tracker to worst match
            for i_det in det_list[order]:
                
                if dists.shape[1] == 0:
                    continue

                # Extract face index with the minimal distance (`min_face`) ...
                min_dist, min_face = dists[i_det, :].min(dim=0)

                # And check whether it is acceptably small
                if min_dist < dist_threshold:

                    # Assign to face_assigned; note that we cannot use `min_face`
                    # directly, because we'll slice `dists` below
                    face_assigned[i_det] = track_list[min_face]

                    # Now, for some magic: remove the selected face
                    # from dists (and track_list), which will make sure
                    # that the next detection cannot be assigned the same
                    # face                    
                    keep = track_list != track_list[min_face]
                    dists = dists[:, keep]
                    track_list = track_list[keep]

            # Update the tracker with the (assigned) detected faces
            unassigned = face_assigned == -1
            tracker[face_assigned[~unassigned]] = lms[~unassigned]

            # If there are new faces, add them to the tracker
            n_new = unassigned.sum()
            if n_new > 0:
                # Update the assigned face index for the new faces with a new integer
                face_assigned[unassigned] = tracker.shape[0] + torch.arange(n_new, device=self.device)
                # and add to tracker
                tracker = torch.cat((tracker, lms[unassigned]))

            # Add face selection to face_idx across images
            face_idx[det_idx] = face_assigned

        self.face_idx = face_idx

        # Loop over unique faces tracked        
        for f in face_idx.unique():
            # Compute the proportion of images containing this face
            f_idx = self.face_idx == f
            prop = (f_idx).sum().div(len(face_idx))

            # Remove this face from each attribute if not present for more than
            # `present_threshold` proportion of images
            if prop < present_threshold:
                for attr in ['conf', 'bbox', 'lms', 'img_idx', 'face_idx']:
                    setattr(self, attr, getattr(self, attr)[~f_idx])

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
                colors = []
                for i_face in self.face_idx[idx].cpu().numpy().astype(int):
                    colors.append(cm.Set1(i_face, bytes=True))
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
