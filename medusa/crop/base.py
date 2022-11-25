import torch
from pathlib import Path
from torchvision.utils import draw_keypoints, save_image


class BaseCropModel:

    def visualize(self, results, f_out):

        if not isinstance(f_out, Path):
            f_out = Path(f_out)

        f_out_base = str(f_out.with_suffix(''))
        f_out_ext = f_out.suffix

        img_crop = results['img_crop']
        n_images = img_crop.shape[0]
        
        img_idx = results['idx']
        for i_img in range(n_images):
            idx = img_idx == i_img
            
            if not torch.any(idx):
                continue

            img = img_crop[idx]
            
            f_out = Path(f_out_base + f'_img-{i_img+1}{f_out_ext}')
            if f_out.is_file():
                f_out.unlink()

            lms = results['lms'][~torch.isnan(img_idx)]
            if lms.ndim == 4:
                lms = lms.squeeze(0)

            img = img.to(torch.uint8)
            for i_det in range(img.shape[0]):
                img = draw_keypoints(img[i_det], lms[None, i_det, :], colors=(0, 255, 0), radius=2)

            save_image(img.float(), fp=f_out, normalize=True)
