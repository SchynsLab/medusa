import cv2
import torch
import numpy as np
from pathlib import Path


class BaseCropModel:

    def _stack_detections(self, images, attrs):
        
        # Total number of detections across all images
        n_det = sum([attr.shape[0] for attr in attrs if attr is not None]) 

        if n_det == 0:
            # No faces detected in any of the images
            return None, None, None

        # Index representing the image belonging to each detection
        idx = torch.zeros(n_det, device=self.device).long()

        # Awkward way to find the shape of the attribute, e.g.
        # (5, 2) for landmarks or (4) for bbox
        attr_shape = None
        for attr in attrs:
            if attr is not None:
                # get shape[1:] because ignore n_faces dim
                attr_shape = attr.shape[1:]
                break

        # batch of: n_detections x 5 x 2 (or n_detections x 4)
        attrs_stacked = torch.zeros((n_det, *attr_shape), device=self.device)

        i = 0
        for i_img, attr in enumerate(attrs):
            
            if attr is None:
                # for this image, no face was detected
                continue
            else:
                # number of detections for this image
                n_det_ = attr.shape[0]

                # assign `i_img` index to this number of detections
                idx[i:(i + n_det_)] = i_img

                # convert to torch if numpy
                if isinstance(attr, np.ndarray):
                    attr = torch.from_numpy(attr).to(self.device)

                # Save attributes
                attrs_stacked[i:(i + n_det_), ...] = attr
                i += n_det_

        # Index original images to get stacked version
        images_stacked = images[idx, ...]

        return images_stacked, attrs_stacked, idx

    def _unstack_detections(self, idx, n_img, *args):
        
        # args_ keeps track of the "unstacked" versions of *args
        args_ = []

        # loop over an arbitrary number of inputs
        for attr in args:
            # container to keep track of unstack version for this attr
            attr_ = []

            # loop over the number of *original* (not stacked!) images,
            # and check whether is is represented in `idx`
            for i_img in range(n_img):
                if i_img not in idx:
                    # this image did not have any detections!
                    attr_.append(None)
                else:
                    # tmp represents the attr instances associated with
                    # the `i_img`th image
                    tmp = attr[idx == i_img, ...]

                    # make sure that we don't lose the batch dim!
                    if tmp.ndim == (attr.ndim - 1):
                        tmp = tmp.unsqueeze(0)

                    attr_.append(tmp)
            
            args_.append(attr_)

        if len(args_) == 1:
            args_ = args_[0]

        return args_

    def visualize(self, img_crop, lmks, f_out):

        n_img = len(img_crop)
        if n_img != len(lmks):
            raise ValueError(f"Number of images ({n_img}) is not the same as "
                             f"the number of lmks ({len(lmks)})!")
        
        if f_out is None:
            to_return = []

        if hasattr(self, 'template'):
            template = self.template.cpu().numpy().round().astype(int)

        for i_img in range(n_img):
            
            img_ = img_crop[i_img]
            if img_ is None:
                return

            lmk_ = lmks[i_img]
            n_det = img_.shape[0]
            for i_det in range(n_det):
                img__ = img_[i_det, ...].permute(1, 2, 0).cpu().numpy()
                img__ = img__[:, :, [2, 1, 0]].copy()  # RGB -> BGR
                lmk__ = lmk_[i_det, ...].cpu().numpy().round().astype(int)

                for i_lmk in range(lmk__.shape[0]):  # probably 5 lms
                    cv2.circle(img__, lmk__[i_lmk, :], 3, (0, 255, 0), -1)
                    if hasattr(self, 'template'):
                        cv2.circle(img__, template[i_lmk, :], 3, (255, 0, 0), -1)
                    
                if n_img == 1 and n_det == 1:
                    idf = ''
                elif n_img == 1 and n_det > 1:
                    idf = f'_det-{i_det:02d}'
                elif n_img > 1 and n_det == 1:
                    idf = f'_img-{i_img:02d}'
                else:
                    idf = f'_img-{i_img:02}_det-{i_det:02d}'

                this_f_out = f_out.parent / (f_out.stem + f'{idf}{f_out.suffix}')

                if this_f_out.is_file():
                    this_f_out.unlink()

                # Note to self: image_ is already BGR (see top of loop)
                success = cv2.imwrite(str(this_f_out), img__)
                if not success:
                    raise ValueError(f"Something went wrong trying to save {str(this_f_out)}!")

        if f_out is None:
            return to_return