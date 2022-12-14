import cv2
import pytest
import numpy as np
from pathlib import Path
from medusa.recon import Mediapipe
from medusa.render import Renderer


imgs = ['no_face', 'one_face', 'two_faces', 'three_faces',
        ['no_face', 'one_face'], ['no_face', 'two_faces'],
        ['one_face', 'two_faces', 'three_faces']]
n_exp = [0, 1, 2, 3, 1, 2, 6]
img_params = zip(imgs, n_exp)


@pytest.mark.parametrize('imgs,n_exp', img_params)
def test_mediapipe_recon(imgs, n_exp):
    if not isinstance(imgs, list):
        imgs = [imgs]

    imgs_path = [Path(__file__).parent / f'test_data/{img}.jpg'
                 for img in imgs]

    recon_model = Mediapipe(static_image_mode=True)
    out = recon_model(imgs_path)

    if n_exp == 0:
        assert(len(out['v']) == 0)
        assert(len(out['mat']) == 0)
    else:
        assert(out['v'].shape[1:] == (468, 3))
        assert(out['mat'].shape[1:] == (4, 4))

    if len(imgs_path) == 1 and len(out['v']) > 0:
        cam_mat = np.eye(4)
        img = cv2.imread(str(imgs_path[0]))[:, :, [2, 1, 0]]
        img_size = img.shape[:2]
        renderer = Renderer(viewport=img_size[::-1], smooth=False, cam_mat=cam_mat, camera_type='intrinsic')

        v = out['v'][0, ...].cpu().numpy()
        recon_img = renderer(v, recon_model.get_tris())
        img = renderer.alpha_blend(recon_img, img)
        f_out = Path(__file__).parent / f'test_viz/recon/test_mediapipe_exp-{n_exp}.png'
        cv2.imwrite(str(f_out), img[:, :, [2, 1, 0]])

        renderer.close()
