import cv2
import pytest
import numpy as np
from pathlib import Path
from medusa.recon import Mediapipe
from medusa.render import Renderer


@pytest.mark.parametrize('imgs_test', [0, 1, 2, 3, 4, [0, 1], [0, 1, 2], [0, 1, 2, 3, 4]], indirect=True)
def test_mediapipe_recon(imgs_test):
    imgs, n_exp = imgs_test
    recon_model = Mediapipe(static_image_mode=True, min_detection_confidence=0.1)
    out = recon_model(imgs)

    if n_exp == 0:
        assert(len(out['v']) == 0)
        assert(len(out['mat']) == 0)
        return
    else:
        assert(out['v'].shape[1:] == (468, 3))
        assert(out['mat'].shape[1:] == (4, 4))

    recon_model.close()

    # Save image of recon when it's just a single image
    # (for visual inspection)
    if isinstance(imgs, Path):

        cam_mat = np.eye(4)
        img = cv2.imread(str(imgs))[:, :, [2, 1, 0]]
        img_size = img.shape[:2]
        renderer = Renderer(viewport=img_size[::-1], shading='flat', cam_mat=cam_mat, cam_type='intrinsic')

        recon_img = renderer(out['v'], recon_model.get_tris())
        img = renderer.alpha_blend(recon_img, img)
        f_out = Path(__file__).parent / f'test_viz/recon/test_mediapipe_exp-{n_exp}.png'
        cv2.imwrite(str(f_out), img[:, :, [2, 1, 0]])

        renderer.close()
