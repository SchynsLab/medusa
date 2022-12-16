import torch
from pathlib import Path
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import warp_affine
from matplotlib import cm
from torchvision.utils import draw_bounding_boxes, draw_keypoints, save_image
from torchvision.ops import box_area

from .. import DEVICE, FONT
from ..io import load_inputs, VideoWriter
from ..log import get_logger


class BatchResults:

    def __init__(self, n_img=0, device=DEVICE, loglevel='INFO', **kwargs):

        self.device = device
        self.n_img = n_img
        self._logger = get_logger(loglevel)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add(self, **kwargs):

        for key, value in kwargs.items():

            # Don't update n_img yet
            if key == 'n_img':
                continue

            existing = getattr(self, key, None)
            if key == 'img_idx':
                # e.g., [0, 0, 1, 2, 2, 3] -> [32, 32, 33, 34, 34, 35]
                # (assuming this is the second batch of 32 images)
                value = value + self.n_img

            if existing is None:
                existing = []

            existing.append(value)
            setattr(self, key, existing)

        # n_img must always be updated last!
        if 'n_img' in kwargs:
            self.n_img += kwargs['n_img']

    def concat(self, n_max=None):

        for attr, data in self.__dict__.items():
            if attr[0] == '_':
                continue

            if attr in ('device', 'n_img'):
                continue

            data = [d for d in data if d is not None]
            if len(data) == 0:
                self._logger.warning(f"No data to concatenate for attribute {attr}!")
                data = None
            else:
                data = torch.cat(data)
                if n_max is not None:
                    data = data[:n_max, ...]

            setattr(self, attr, data)

    def sort_faces(self, attr='lms', dist_threshold=250, present_threshold=0.1):

        if not hasattr(self, attr):
            raise ValueError(f"Cannot sort faces using attribute `{attr}`, which does not exist!")

        data = getattr(self, attr, None)
        if data is None:
            raise ValueError(f"Cannot sort faces using attribut `{attr}`, because it's empty!")

        img_idx = getattr(self, 'img_idx', None)
        if img_idx is None:
            raise ValueError(f"Cannot sort faces because `img_idx` is not known!")

        if getattr(self, 'face_idx', None) is not None:
            raise ValueError("`face_idx` already computed!")

        self.face_idx, keep = sort_faces(data, img_idx, dist_threshold, present_threshold)
        if not torch.all(keep):
            for attr, data in self.to_dict(exclude=['n_img', 'device']).items():
                if data.shape[0] == keep.shape[0]:
                    setattr(self, attr, data[keep])

    def to_dict(self, exclude=None):

        to_return = self.__dict__.copy()
        for attr in to_return.copy():
            if attr[0] == '_':
                del to_return[attr]

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]

            for to_exclude in exclude:
                if to_exclude in to_return:
                    del to_return[to_exclude]

        return to_return

    def visualize(self, f_out, imgs, video=False, show_cropped=False, face_id=None, fps=24, crop_size=(224, 224), template=None, **kwargs):
        """Creates an image with the estimated bounding box (bbox) on top of
        it.

        Parameters
        ----------
        f_out : str, pathlib.Path
            If multiple images, a number (_xxx) is appended
        imgs : array_like
            A numpy array with the original (uncropped images); can also be
            a torch Tensor; can be a batch of images or a single image
        bbox : np.ndarray
            A numpy array with the bounding box(es) corresponding to the
            image(s)
        """
        # To be used in recursive function call
        recursive_kwargs = locals()
        if not isinstance(f_out, Path):
            f_out = Path(f_out)

        if f_out.is_file():
            # Remove if exists already
            f_out.unlink()

        # batch_size x h x w x 3
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=self.device, dtype='uint8')

        if show_cropped and face_id is None:
            # Going into recursive loop, looping over each unique face in the
            # set of detections across images
            n_face = self.face_idx.unique().numel()
            for face_id in self.face_idx.unique():
                # If there's just one face, use the original filename;
                # otherwise, append a unique identifier (_face-xx)
                if n_face > 1:
                    recursive_kwargs['f_out'] = Path(str(f_out.with_suffix('')) + f'_face-{face_id+1}.mp4')

                # Set current face id in inputs
                recursive_kwargs['face_id'] = face_id
                recursive_kwargs.pop('self', None)  # don't need this

                # Recursive call, but set face_id
                self.visualize(**recursive_kwargs)

            # If we have arrived here, we have finished our recursive loop and can safely
            # exit the function
            return

        imgs_out = []

        # Loop over all images (in the video)
        n_images = imgs.shape[0]
        for i_img in range(n_images):
            img = imgs[i_img, ...]

            # If we don't know the `img_idx`, just save the
            # original image
            if getattr(self, 'img_idx', None) is None:
                imgs_out.append(img.unsqueeze(0))
                continue

            # If the current image has no detections, just
            # reutrn the original image
            if i_img not in self.img_idx:
                if show_cropped:
                    # Don't need to save the original image, because
                    # there's no way to warp it to cropped image space
                    # without any detection
                    continue

                imgs_out.append(img.unsqueeze(0))
                continue

            # Determine which detections are in this image (det_idx)
            det_idx = self.img_idx == i_img

            if show_cropped:
                # If we want to show the cropped image instead of the original
                # image, we additionally need to check whether the current `face_id`
                # is detected in the current image (`img`)
                if face_id not in self.face_idx[det_idx]:
                    # This face is not detected in this image!
                    continue

                # We only want to visualize the detection from the current face
                # (`self.face_idx == face_id`) so update `det_idx`
                det_idx = torch.logical_and(det_idx, self.face_idx == face_id)

            # We only want to show the bounding box for uncropped images
            # (because for cropped images, borders = bbox)
            if hasattr(self, 'bbox') and not show_cropped:
                bbox = self.bbox[det_idx]

                # Check for confidence of detection, which
                # we'll draw if available
                if hasattr(self, 'conf'):
                    # Heuristic for scaling font size
                    font_size = int((box_area(bbox).min().sqrt() / 8).item())
                    labels = [str(round(lab.item(), 3)) for lab in self.conf[det_idx]]
                else:
                    labels = None

                # If `face_idx` is available, give each unique face a bounding
                # box with a separate color
                if getattr(self, 'face_idx', None) is not None:
                    colors = []
                    for i_face in self.face_idx[det_idx]:
                        colors.append(cm.Set1(i_face.item(), bytes=True))
                else:
                    colors = (255, 0, 0)

                # TODO: scale width
                img = draw_bounding_boxes(img, bbox, labels, colors, width=2, font=FONT, font_size=font_size)
                img = img.to(self.device)

            # BELOW: OLD CODE TO CREATE BOUNDING BOX FROM CROPPED IMAGES
            # bbox_crop = torch.tensor([[0, 0], [0, h-1], [h-1, w-1], [0, w-1]], dtype=torch.float32, device=self.device)
            # bbox_crop = bbox_crop.repeat(b, 1, 1)
            # crop_mats = torch.inverse(self.crop_mats[idx])
            # bbox = transform_points(crop_mats, bbox_crop)

            # Check for landmarks (`lms`), which we'll draw if available
            if hasattr(self, 'lms'):
                lms = self.lms[det_idx]

                if show_cropped:
                    # Need to crop the original images!
                    crop_mats = self.crop_mats[det_idx]
                    img = warp_affine(img.unsqueeze(0).float(), crop_mats[:, :2, :], crop_size)
                    img = img.to(torch.uint8).squeeze(0)

                    # And warp the landmarks to the cropped image space
                    lms = transform_points(crop_mats, lms)

                # TODO: scale radius
                img = draw_keypoints(img, lms, colors=(0, 255, 0), radius=2)
                img = img.to(self.device)

                if template is not None:
                    if show_cropped:
                        template_ = template.unsqueeze(0)
                    else:
                        crop_mats = torch.inverse(self.crop_mats[det_idx])
                        template_ = template.repeat(lms.shape[0], 1, 1).to(crop_mats.device)
                        template_ = transform_points(crop_mats, template_)

                    img = draw_keypoints(img, template_, colors=(0, 0, 255), radius=1.5)
                    img = img.to(self.device)

            # Add back batch dimension (so we can concatenate later)
            imgs_out.append(img.unsqueeze(0))

        # batch_dim x 3 x h x w
        imgs_out = torch.cat(imgs_out)

        if video:
            writer = VideoWriter(str(f_out), fps=fps)
            writer.write(imgs_out.permute(0, 2, 3, 1))
            writer.close()
        else:
            save_image(imgs_out.float(), fp=f_out, normalize=True, **kwargs)


def sort_faces(lms, img_idx, dist_threshold=250, present_threshold=0.1):
    device = lms.device
    face_idx = torch.zeros_like(img_idx, device=device, dtype=torch.int64)

    for i, i_img in enumerate(img_idx.unique()):
        # lms = all detected landmarks/vertices for this image
        det_idx = i_img == img_idx
        lms_img = lms[det_idx]

        # flatten landmarks (5 x 2 -> 10)
        n_det = lms_img.shape[0]
        lms_img = lms_img.reshape((n_det, -1))

        if i == 0:
            # First detection, initialize tracker with first landmarks
            face_idx[det_idx] = torch.arange(0, n_det, device=device)
            tracker = lms_img.clone()
            continue

        # Compute the distance between each detection (lms) and currently tracked faces (tracker)
        dists = torch.cdist(lms_img, tracker)  # n_det x current_faces

        # face_assigned keeps track of which detection is assigned to which face from
        # the tracker (-1 refers to "not assigned yet")
        face_assigned = torch.ones(n_det, device=device, dtype=torch.int64) * -1

        # We'll keep track of which tracked faces we have not yet assigned
        track_list = torch.arange(dists.shape[1], device=device)

        # Check the order of minimum distances across detections
        # (which we'll use to loop over)
        order = dists.min(dim=1)[0].argsort()
        det_list = torch.arange(dists.shape[0], device=device)

        # Loop over detections, sorted from best match (lowest dist)
        # to any face in the tracker to worst match
        for i_det in det_list[order]:

            if dists.shape[1] == 0:
                # All faces from tracker have been assigned!
                # So this detection must be a new face
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
            else:
                # Detection cannot be assigned to any face in the tracker!
                # Going to update tracker with this detection
                pass

        # Update the tracker with the (assigned) detected faces
        unassigned = face_assigned == -1
        tracker[face_assigned[~unassigned]] = lms_img[~unassigned]

        # If there are new faces, add them to the tracker
        n_new = unassigned.sum()
        if n_new > 0:
            # Update the assigned face index for the new faces with a new integer
            face_assigned[unassigned] = tracker.shape[0] + torch.arange(n_new, device=device)
            # and add to tracker
            tracker = torch.cat((tracker, lms_img[unassigned]))

        # Add face selection to face_idx across images
        face_idx[det_idx] = face_assigned

    # Loop over unique faces tracked
    keep = torch.full_like(face_idx, fill_value=True, dtype=torch.bool)
    for f in face_idx.unique():
        # Compute the proportion of images containing this face
        f_idx = face_idx == f
        prop = (f_idx).sum().div(len(face_idx))
        if prop < present_threshold:
            keep[f_idx] = False

    return face_idx, keep
