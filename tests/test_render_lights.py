from pathlib import Path
from medusa.crop import BboxCropModel
from medusa.data import get_example_image
from medusa.recon import DecaReconModel
from medusa.render import PytorchRenderer
from medusa.render.lights import SphericalHarmonicsLights


def test_spherical_harmonics_lights():

    img = get_example_image(load=True)
    recon_model = DecaReconModel("emoca-coarse", orig_img_size=(img.shape[3], img.shape[2]))
    crop_model = BboxCropModel()
    out_crop = crop_model(img)
    img = recon_model._preprocess(out_crop['imgs_crop'])
    enc_dict = recon_model._encode(img)
    dec_dict = recon_model._decode(enc_dict, out_crop['crop_mat'])

    viewport = (img.shape[3], img.shape[2])
    cam_mat = recon_model.get_cam_mat()
    tris = recon_model.get_tris()
    lights = SphericalHarmonicsLights(enc_dict['light'])

    renderer = PytorchRenderer(viewport, cam_mat=cam_mat, cam_type="orthographic",
                               shading='flat', lights=lights)
    img_r = renderer(dec_dict['v'], tris)
    f_out = Path(__file__).parent / "test_viz/render/test_spherical_harmonics.jpg"
    renderer.save_image(f_out, img_r)
