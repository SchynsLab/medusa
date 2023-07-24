"""A very hack implementation of a container to store results from processing
multiple batches of images."""

from pathlib import Path

import torch
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import warp_affine
from matplotlib import cm
from torchvision.ops import box_area
from torchvision.utils import draw_bounding_boxes, draw_keypoints, save_image

from ..defaults import DEVICE, FONT, LOGGER
from ..io import VideoWriter, load_inputs
from ..tracking import _ensure_consecutive_face_idx, filter_faces, sort_faces


class BatchResults:
    """A container to store and process results from processing multiple
    batches of inputs/images.

    Parameters
    ----------
    n_img : int
        Number of images processed thus far
    device : str
        Device to store/process the data on (either 'cpu' or 'cuda')
    **kwargs
        Other data that will be set as attributes
    """

    def __init__(self, n_img=0, device=DEVICE, **kwargs):
        """Initializes a BatchResults object."""
        self.device = device
        self.n_img = n_img

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add(self, **kwargs):
        """Add data to the container.

        Parameters
        ----------
        **kwargs
            Any data that will be added to the
        """
        for key, value in kwargs.items():

            # Don't update n_img yet
            if key == "n_img":
                continue

            existing = getattr(self, key, None)
            if key == "img_idx":
                # e.g., [0, 0, 1, 2, 2, 3] -> [32, 32, 33, 34, 34, 35]
                # (assuming this is the second batch of 32 images)
                value = value + self.n_img

            if existing is None:
                existing = []

            existing.append(value)
            setattr(self, key, existing)

        # n_img must always be updated last!
        if "n_img" in kwargs:
            self.n_img += kwargs["n_img"]

    def concat(self, n_max=None):
        """Concatenate results form multiple batches.

        Parameters
        ----------
        n_max : None, int
            Whether to only return ``n_max`` observations per attribute
            (ignored if ``None``)
        """
        for attr, data in self.__dict__.items():
            if attr[0] == "_":
                continue

            if attr in ("device", "n_img"):
                continue

            data = [d for d in data if d is not None]
            if len(data) == 0:
                LOGGER.warning(f"No data to concatenate for attribute {attr}!")
                data = None
            else:
                data = torch.cat(data)
                if n_max is not None:
                    data = data[:n_max, ...]

            setattr(self, attr, data)

    def sort_faces(self, attr="lms", dist_threshold=250):
        """'Sorts' faces using the ``medusa.tracking.sort_faces`` function (and
        performs some checks of the data).

        Parameters
        ----------
        attr : str
            Name of the attribute that needs to be used to sort the faces
            (e.g., 'lms' or 'v')
        dist_threshold : int, float
            Euclidean distance between two sets of landmarks/vertices that we consider
            comes from two different faces (e.g., if ``d(lms1, lms2) >= dist_treshold``,
            then we conclude that face 1 (``lms1``) is a different from face 2 (``lms2``)

        Returns
        -------
        face_idx : torch.tensor
            The face IDs associate with each detection
        """
        if not hasattr(self, attr):
            LOGGER.warning(f"No attribute `{attr}`, maybe no detections?")
            return

        data = getattr(self, attr, None)
        if data is None:
            raise ValueError(
                f"Cannot sort faces using attribut `{attr}`, because it's empty!"
            )

        img_idx = getattr(self, "img_idx", None)
        if img_idx is None:
            raise ValueError(f"Cannot sort faces because `img_idx` is not known!")

        if getattr(self, "face_idx", None) is not None:
            raise ValueError("`face_idx` already computed!")

        self.face_idx = sort_faces(data, img_idx, dist_threshold)

    def filter_faces(self, present_threshold=0.1):

        keep = filter_faces(self.face_idx, self.n_img, present_threshold)

        if not torch.all(keep):
            for attr, data in self.to_dict(exclude=["n_img", "device"]).items():
                if data.shape[0] == keep.shape[0]:
                    setattr(self, attr, data[keep])

        self.face_idx = _ensure_consecutive_face_idx(self.face_idx)

    def to_dict(self, exclude=None):

        to_return = self.__dict__.copy()
        for attr in to_return.copy():
            if attr[0] == "_":
                del to_return[attr]

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]

            for to_exclude in exclude:
                if to_exclude in to_return:
                    del to_return[to_exclude]

        return to_return

    def visualize(
        self,
        f_out,
        imgs,
        video=False,
        show_cropped=False,
        face_id=None,
        fps=24,
        crop_size=(224, 224),
        template=None,
        **kwargs,
    ):
        """Visualizes the detection/cropping results aggregated by the
        BatchResults object.

        Parameters
        ----------
        f_out : str, Path
            Path of output image/video
        imgs : torch.tensor
            A tensor with the original (uncropped images); can be a batch of images
            or a single image
        video : bool
            Whether to output a video or image (grid)
        show_cropped : bool
            Whether to visualize the cropped image or the original image
        face_id : None
            Should be None (used in recursive call)
        fps : int
            Frames per second of video (only relevant if ``video=True``)
        crop_size : tuple[int]
            Size of cropped images
        template : torch.tensor
            Template used in aligment (optional)
        """
        # To be used in recursive function call
        recursive_kwargs = locals()
        if not isinstance(f_out, Path):
            f_out = Path(f_out)

        if f_out.is_file():
            # Remove if exists already
            f_out.unlink()

        # batch_size x h x w x 3
        imgs = load_inputs(
            imgs,
            load_as="torch",
            channels_first=True,
            device=self.device,
            dtype="uint8",
        )

        if show_cropped and face_id is None:
            # Going into recursive loop, looping over each unique face in the
            # set of detections across images
            n_face = self.face_idx.unique().numel()
            for face_id in self.face_idx.unique():
                # If there's just one face, use the original filename;
                # otherwise, append a unique identifier (_face-xx)
                if n_face > 1:
                    recursive_kwargs["f_out"] = Path(
                        str(f_out.with_suffix("")) + f"_face-{face_id+1}{f_out.suffix}"
                    )

                # Set current face id in inputs
                recursive_kwargs["face_id"] = face_id
                recursive_kwargs.pop("self", None)  # don't need this

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
            if getattr(self, "img_idx", None) is None:
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
            if hasattr(self, "bbox") and not show_cropped:
                bbox = self.bbox[det_idx]
                font_size = int((box_area(bbox).min().sqrt() / 8).item())

                # Check for confidence of detection, which
                # we'll draw if available
                if hasattr(self, "conf"):
                    # Heuristic for scaling font size
                    labels = [str(round(lab.item(), 3)) for lab in self.conf[det_idx]]
                else:
                    labels = None

                # If `face_idx` is available, give each unique face a bounding
                # box with a separate color
                if getattr(self, "face_idx", None) is not None:
                    colors = []
                    for i_face in self.face_idx[det_idx]:
                        colors.append(cm.Set1(i_face.item(), bytes=True))
                else:
                    colors = (255, 0, 0)

                # TODO: scale width
                img = draw_bounding_boxes(
                    img, bbox, labels, colors, width=2, font=FONT, font_size=font_size
                )
                img = img.to(self.device)

            # BELOW: OLD CODE TO CREATE BOUNDING BOX FROM CROPPED IMAGES
            # bbox_crop = torch.tensor([[0, 0], [0, h-1], [h-1, w-1], [0, w-1]], dtype=torch.float32, device=self.device)
            # bbox_crop = bbox_crop.repeat(b, 1, 1)
            # crop_mat = torch.inverse(self.crop_mat[idx])
            # bbox = transform_points(crop_mat, bbox_crop)

            # Check for landmarks (`lms`), which we'll draw if available
            if hasattr(self, "lms"):
                lms = self.lms[det_idx]

                if show_cropped:
                    # Need to crop the original images!
                    crop_mat = self.crop_mat[det_idx]
                    img = warp_affine(
                        img.unsqueeze(0).float(), crop_mat[:, :2, :], crop_size
                    )
                    img = img.to(torch.uint8).squeeze(0)

                    # And warp the landmarks to the cropped image space
                    lms = transform_points(crop_mat, lms)

                # TODO: scale radius
                img = draw_keypoints(img, lms, colors=(0, 255, 0), radius=2)
                img = img.to(self.device)

                if template is not None:
                    if show_cropped:
                        template_ = template.unsqueeze(0)
                    else:
                        crop_mat = torch.inverse(self.crop_mat[det_idx])
                        template_ = template.repeat(lms.shape[0], 1, 1).to(
                            crop_mat.device
                        )
                        template_ = transform_points(crop_mat, template_)

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
