from pathlib import Path

import torch
from kornia.geometry.linalg import transform_points
from kornia.utils import draw_line
from matplotlib import cm
from torchvision.io import write_video
from torchvision.utils import draw_bounding_boxes, draw_keypoints, save_image

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

    @classmethod
    def from_batches(cls, batches):

        imgs_crop = torch.concatenate([b.imgs_crop for b in batches if b.imgs_crop is not None])
        crop_mats = torch.concatenate([b.crop_mats for b in batches if b.crop_mats is not None])
        lms = torch.concatenate([b.lms for b in batches if b.lms is not None])

        n_img = batches[0].n_img
        for i in range(1, len(batches)):
            if batches[i].img_idx is not None:
                batches[i].img_idx += n_img
            n_img += batches[i].n_img

        img_idx = torch.concatenate([b.img_idx for b in batches if b.img_idx is not None])
        device = batches[-1].device

        # Note to self: we can safely ignore face_idx is it exists already,
        # because it is probably not consistent across batches anyway
        return cls(n_img, imgs_crop, crop_mats, lms, img_idx, None, device)

    def sort(self, dist_threshold=200, present_threshold=0.1):

        if self.face_idx is not None:
            # Maybe a warning or error here?
            pass

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

    def visualize(self, imgs, f_out, template=None, show_crop=False, video=False, **kwargs):

        if show_crop and self.face_idx is None:
            raise ValueError("Cannot visualize cropped images without knowing "
                             "which faces belong to which crops; run `sort` first!")

        if not isinstance(f_out, Path):
            f_out = Path(f_out)

        if f_out.is_file():
            f_out.unlink()

        if show_crop:
            n_face = self.face_idx.unique().numel()
            for i_face in self.face_idx.unique():
                idx = self.face_idx == i_face
                imgs = self.imgs_crop[idx].to(torch.uint8)
                lms = transform_points(self.crop_mats[idx], self.lms[idx])

                for i_img in range(imgs.shape[0]):
                    imgs[i_img] = draw_keypoints(imgs[i_img], lms[[i_img]], colors=(0, 255, 0), radius=1.5)

                    if template is not None:
                        imgs[i_img] = draw_keypoints(imgs[i_img], template.unsqueeze(0), colors=(0, 0, 255), radius=1.5)

                if n_face != 1:
                    this_f_out = Path(str(f_out.with_suffix('')) + f'_face-{i_face+1}.mp4')
                else:
                    this_f_out = f_out

                if this_f_out.is_file():
                    this_f_out.unlink()

                if video:
                    write_video(str(this_f_out), imgs.permute(0, 2, 3, 1).cpu(), fps=24)
                else:
                    save_image(imgs.float(), fp=this_f_out, normalize=True, **kwargs)

            return

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
            bbox_crop = torch.tensor([[0, 0], [0, h-1], [h-1, w-1], [0, w-1]], dtype=torch.float32, device=self.device)
            bbox_crop = bbox_crop.repeat(b, 1, 1)
            crop_mats = torch.inverse(self.crop_mats[idx])
            bbox = transform_points(crop_mats, bbox_crop)
            # maxx = torch.tensor([img.shape[2], img.shape[1]], device=self.device)
            # bbox = torch.clamp(bbox, min=torch.zeros(2, device=self.device), max=maxx - 1).to(torch.int64)
            # green = torch.tensor([0, 255, 0], device=self.device)[:, None]
            # img = img.to(torch.int64)
            # for i_box in range(bbox.shape[0]):
            #     bbox_ = bbox[i_box, :, :]
            #     img = draw_line(img, bbox_[0, :], bbox_[1, :], color=green)
            #     img = draw_line(img, bbox_[1, :], bbox_[2, :], color=green)
            #     img = draw_line(img, bbox_[0, :], bbox_[3, :], color=green)
            #     img = draw_line(img, bbox_[3, :], bbox_[2, :], color=green)
            # img = img.to(torch.uint8)
            bbox = bbox[:, [0, 2], :].reshape((b, 4))
            img = draw_bounding_boxes(img, bbox, width=2, labels=None, colors=colors)

            # Also draw landmarks (which should be already in original image space)
            img = draw_keypoints(img, self.lms[idx], colors=(0, 255, 0), radius=1.5)

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
