import torch
from pathlib import Path
from torchvision.utils import draw_bounding_boxes, draw_keypoints, save_image

from .. import DEVICE, FONT
from ..io import load_inputs


class BaseDetectionModel:

    @staticmethod
    def visualize(imgs, results, f_out, device=DEVICE):
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

        f_out_base = str(f_out.with_suffix(''))
        f_out_ext = f_out.suffix

        # batch_size x h x w x 3
        imgs = load_inputs(imgs, load_as='torch', channels_first=True, device=DEVICE)
        n_images = imgs.shape[0]
        
        img_idx = results['idx']
        for i_img in range(n_images):
            img = imgs[i_img, ...]
            idx = img_idx == i_img

            f_out = Path(f_out_base + f'_img-{i_img+1}{f_out_ext}')
            if f_out.is_file():
                f_out.unlink()

            if not torch.any(idx):
                save_image(img, fp=f_out, normalize=True)  # empty
                continue

            bbox = results['bbox'][idx, ...]
            if bbox.ndim == 3:
                bbox = bbox.squeeze(0)
            
            labels = [str(lab) for lab in results['conf'][idx].cpu().numpy().round(3)]
            img = draw_bounding_boxes(img.to(torch.uint8), bbox, labels, (255, 0, 0), font=FONT, font_size=20)

            lms = results['lms'][idx, ...]
            if lms.ndim == 4:
                lms = lms.squeeze(0)

            img = draw_keypoints(img, lms, colors=(0, 255, 0), radius=2)
            save_image(img.float(), fp=f_out, normalize=True)
