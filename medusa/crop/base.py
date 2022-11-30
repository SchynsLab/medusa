import torch
from pathlib import Path
from torchvision.utils import draw_keypoints, save_image, draw_bounding_boxes
from torchvision.io import write_video
from kornia.geometry.linalg import transform_points
from matplotlib import cm

from .. import DEVICE
from ..io import load_inputs
from ..tracking import sort_faces

class BaseCropModel:
    pass


class CropResults:

    def __init__(self, n_img, imgs_crop=None, crop_mats=None, lms=None, img_idx=None, face_idx=None, device=DEVICE):
        self.n_img = n_img
        self.imgs_crop = imgs_crop
        self.crop_mats = crop_mats
        self.lms = lms
        self.img_idx = img_idx
        self.face_idx = face_idx
        self.device = device

    def sort(self, dist_threshold=200, present_threshold=0.1):
        
        self.face_idx = sort_faces(self.lms, self.img_idx, dist_threshold, self.device)
        
        # Loop over unique faces tracked        
        for f in self.face_idx.unique():
            # Compute the proportion of images containing this face
            f_idx = self.face_idx == f
            prop = (f_idx).sum().div(len(self.face_idx))

            # Remove this face from each attribute if not present for more than
            # `present_threshold` proportion of images
            if prop < present_threshold:
                for attr in ['imgs_crop', 'crop_mats', 'lms', 'img_idx', 'face_idx']:
                    setattr(self, attr, getattr(self, attr)[~f_idx])

    def visualize(self, imgs, f_out, template=None, video=False, **kwargs):

        if not isinstance(f_out, Path):
            f_out = Path(f_out)

        # batch_size x h x w x 3
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=self.device, dtype='uint8')
        n_images = imgs.shape[0]

        for i_img in range(n_images):
            
            img = imgs[i_img, ...]
            if self.img_idx is None:
                continue
            # or only the current image has no detections
            elif i_img not in self.img_idx:
                continue

            idx = self.img_idx == i_img
            b, c, h, w = self.imgs_crop[idx].shape

            if self.face_idx is not None:
                colors = []
                for i_face in self.face_idx[idx].cpu().numpy().astype(int):
                    colors.append(cm.Set1(i_face, bytes=True))
            else:
                colors = (255, 0, 0)

            # Create the bbox used for cropping by warping top-left and bottom-right corner
            # to the original image space using the previously estimated crop matrix
            corners = torch.tensor([[0, 0], [h-1, w-1]], dtype=torch.float32, device=self.device)
            bbox_crop = corners.repeat(b, 1, 1)
            crop_mats = torch.inverse(self.crop_mats[idx])
            bbox = transform_points(crop_mats, bbox_crop).reshape((b, 4))
            img = draw_bounding_boxes(img, bbox, width=2, labels=None, colors=colors)

            # Also transform landmark points back to the original image space
            # (maybe don't warp them to cropped image space at all?)
            lms = transform_points(crop_mats, self.lms[idx])           
            img = draw_keypoints(img, lms, colors=(0, 255, 0), radius=1.5)

            if template is not None:
                temp = transform_points(crop_mats, template.repeat(b, 1, 1).to(crop_mats.device))
                img = draw_keypoints(img, temp, colors=(0, 0, 255), radius=1.5)
            
            imgs[i_img] = img

        if f_out.is_file():
            # Remove if exists already
            f_out.unlink()

        if video:
            write_video(str(f_out), imgs.permute(0, 2, 3, 1).cpu(), fps=24)            
        else:
            save_image(imgs.float(), fp=f_out, normalize=True, **kwargs)
