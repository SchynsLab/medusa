"""Module with different FLAME-based 3D reconstruction models, including DECA.

[1]_, EMOCA [2]_

All model classes inherit from a common base class, ``FlameReconModel`` (see
``flame.base`` module).

.. [1] Feng, Y., Feng, H., Black, M. J., & Bolkart, T. (2021). Learning an animatable detailed
       3D face model from in-the-wild images. ACM Transactions on Graphics (ToG), 40(4), 1-13.
.. [2] Danecek, R., Black, M. J., & Bolkart, T. (2022). EMOCA: Emotion Driven Monocular
       Face Capture and Animation. arXiv preprint arXiv:2204.11312.

For the associated license, see license.md.
"""

from pathlib import Path

import numpy as np
import torch
from kornia.geometry.linalg import transform_points

from ....defaults import LOGGER
from ....transforms import (create_ortho_matrix, create_viewport_matrix,
                            crop_matrix_to_3d)
from ..base import FlameReconModel
from ..decoders import FlameShape, FlameTex
from .decoders import DetailGenerator
from .encoders import ResnetEncoder
from ....crop import BboxCropModel
from ....defaults import FLAME_MODELS, DEVICE
from ....geometry import compute_vertex_normals


class DecaReconModel(FlameReconModel):
    """A 3D face reconstruction model that uses the FLAME topology.

    At the moment, six different models are supported: 'deca-coarse', 'deca-dense',
    'emoca-coarse', 'emoca-dense'

    Parameters
    ----------
    name : str
        Either 'deca-coarse', 'deca-dense', 'emoca-coarse', or 'emoca-dense'
    extract_lms : None, str
        If ``None``, no landmarks are extracted; if '68', 68 landmarks are extracted;
        if 'mp', 468 mediapipe landmarks are extracted
    orig_img_size : tuple
        Original (before cropping!) image dimensions of video frame (width, height);
        needed for baking in translation due to cropping; if not set, it is assumed
        that the image is not cropped!
    """

    def __init__(self, name, extract_tex=False, orig_img_size=None, device=DEVICE):
        """Initializes an DECA-like model object."""
        super().__init__()
        self.name = name
        self.extract_tex = extract_tex
        self.orig_img_size = orig_img_size
        self.device = device
        self._dense = "dense" in name
        self._warned_about_crop_mat = False
        self._check()
        self._load_data()
        self._create_submodels()
        self._crop_model = BboxCropModel("2d106det", (224, 224), device=device)
        self.to(device)

    def __str__(self):
        return self.name

    def _check(self):
        """Does some checks of the parameters."""

        if self.name not in FLAME_MODELS:
            raise ValueError(f"Name must be in {FLAME_MODELS}, but got {self.name}!")

    def _load_data(self):
        """Loads necessary data."""
        # Avoid circular import
        from ....data import get_external_data_config, get_template_flame

        self._template = {
            'coarse': get_template_flame('coarse', keys=['tris'], device=self.device)
        }
        data_dir = Path(__file__).parents[3] / 'data/flame'

        if self._dense:
            self._template['dense'] = get_template_flame(topo='dense', device=self.device)
            self._fixed_uv_dis = np.load(data_dir / "fixed_displacement_256.npy")
            self._fixed_uv_dis = torch.as_tensor(self._fixed_uv_dis, device=self.device).float()

        self._cfg = get_external_data_config()

    def _create_submodels(self):
        """Creates all EMOCA encoding and decoding submodels. To summarize:

        - ``E_flame``: predicts (coarse) FLAME parameters given an image
        - ``E_expression``: predicts expression FLAME parameters given an image
        - ``E_detail``: predicts detail FLAME parameters given an image
        - ``D_flame``: outputs a ("coarse") mesh given (shape, exp, pose) FLAME parameters
        - ``D_flame_tex``: outputs a texture map given (tex) FLAME parameters
        - ``D_detail``: outputs detail map (in uv space) given (detail) FLAME parameters
        """

        # set up parameter list and dict
        self.param_dict = {
            "n_shape": 100,
            "n_tex": 50,
            "n_exp": 50,
            "n_pose": 6,
            "n_cam": 3,
            "n_light": 27,
        }

        # Flame encoder: image -> n_param FLAME parameters
        n_param = sum([n for n in self.param_dict.values()])
        self.E_flame = ResnetEncoder(outsize=n_param)

        if self._dense:
            # Detail map encoder: image -> 128 detail latents
            self.E_detail = ResnetEncoder(outsize=128)

        if "emoca" in self.name:
            # Expression encoder: image -> 50 expression parameters
            self.E_expression = ResnetEncoder(self.param_dict["n_exp"])

        # Flame decoder: n_param Flame parameters -> vertices / affine matrix
        self.D_flame_shape = FlameShape(n_shape=100, n_expr=50)

        if self.extract_tex:
            self.D_flame_tex = FlameTex(n_tex=self.param_dict['n_tex'])

        if self._dense:
            # Detail decoder: (detail, exp, cam params) -> detail map
            self.D_detail = DetailGenerator(
                latent_dim=(128 + 50 + 3),  # (n_detail, n_exp, n_cam),
                out_channels=1,
                out_scale=0.01,
                sample_mode="bilinear",
            )

        # Load weights from checkpoint and apply to models
        checkpoint = torch.load(self._cfg[self.name.split("-")[0] + "_path"])
        self.E_flame.load_state_dict(checkpoint["E_flame"])

        if self._dense:
            self.E_detail.load_state_dict(checkpoint["E_detail"])
            self.D_detail.load_state_dict(checkpoint["D_detail"])

        if "emoca" in self.name:
            self.E_expression.load_state_dict(checkpoint["E_expression"])
            # for some reason E_exp should be explicitly cast to cuda

    def _encode(self, imgs):
        """"Encodes" the image into FLAME parameters, i.e., predict FLAME
        parameters for the given image. Note that, at the moment, it only works
        for a single image, not a batch of images.

        Parameters
        ----------
        imgs : torch.Tensor
            A Tensor with shape b (batch size) x 3 (color ch.) x 244 (w) x 244 (h)

        Returns
        -------
        enc_dict : dict
            A dictionary with all encoded parameters and some extra data needed
            for the decoding stage.
        """

        if self.orig_img_size is None:
            # If orig_img_size was not set upon initialization, assume no cropping
            # and use the size of the current image
            self.orig_img_size = tuple(imgs.shape[2:])  # h x w

        # Encode image into FLAME parameters, then decompose parameters
        # into a dict with parameter names (shape, tex, exp, etc) as keys
        # and the estimated parameters as values
        enc_params = self.E_flame(imgs)
        enc_dict = self._decompose_params(enc_params, self.param_dict)

        # Note to self:
        # enc_dict['cam'] contains [batch_size, x_trans, y_trans, zoom] (in mm?)
        # enc_dict['pose'] contains [rot_x, rot_y, rot_z] (in radians) for the neck
        # and jaw (first 3 are neck, last three are jaw)
        # rot_x_jaw = mouth opening
        # rot_y_jaw = lower jaw to left or right
        # rot_z_jaw = not really possible?

        # Encode image into detail parameters
        if self._dense:
            detail_params = self.E_detail(imgs)
            enc_dict["detail"] = detail_params

        # Replace "DECA" expression parameters with EMOCA-specific
        # expression parameters
        if "emoca" in self.name:
            enc_dict["exp"] = self.E_expression(imgs)

        return enc_dict

    def _decompose_params(self, parameters, num_dict):
        """Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']."""
        enc_dict = {}
        start = 0

        for key in num_dict:
            key = key[2:]  # trim off 'n_'
            end = start + int(num_dict["n_" + key])
            enc_dict[key] = parameters[:, start:end]
            start = end

            if key == "light":
                # Reshape 27 flattened params into 9 x 3 array
                enc_dict[key] = enc_dict[key].reshape(enc_dict[key].shape[0], 9, 3)

        return enc_dict

    def _decode(self, enc_dict, crop_mat):
        """Decodes the face attributes (vertices, landmarks, texture, detail
        map) from the encoded parameters.

        Parameters
        ----------
        enc_dict : dict
            Dictionary with encoding parameters with keys 'shape', 'exp',
            'pose', 'cam' and 'light'

        Returns
        -------
        dec_dict : dict
            A dictionary with the results from the decoding stage, with keys
            'v' (vertices) and 'mat' (4 x 4 affine matrices)

        Raises
        ------
        ValueError
            If `tform` parameter is not `None` and `orig_size` is `None`. In other
            words, if `tform` is supplied, `orig_size` should be supplied as well,
            otherwise
        """

        # Decode vertices (`v`) and rotation params (`R`) from the shape/exp/pose params
        v, R = self.D_flame_shape(shape=enc_dict['shape'], expr=enc_dict['exp'],
                                  global_pose=enc_dict['pose'][:, :3],
                                  jaw_pose=enc_dict['pose'][:, 3:]
        )
        b = v.shape[0]  # batch dim

        if self._dense:
            input_detail = torch.cat(
                [enc_dict["pose"][:, 3:], enc_dict["exp"], enc_dict["detail"]], dim=1
            )
            uv_z = self.D_detail(input_detail)
            disp_map = uv_z.squeeze(1) + self._fixed_uv_dis[None, :, :]
            normals = compute_vertex_normals(v, self._template['coarse']['tris'])

            # Upsample mesh to 'dense' format given (coarse) vertices and displacement map
            # For now, this is done in numpy format, and cast back to torch Tensor afterwards
            v = self._upsample_mesh(v, normals, disp_map)

        # Note that `v` is in world space, but pose (global rotation only)
        # is already applied
        cam = enc_dict["cam"]

        # Actually, R is per vertex (not sure why) but doesn't really differ
        # across vertices, so let's average
        R = R.mean(axis=1)

        # Now, translation. We are going to do something weird. EMOCA (and
        # DECA) estimate translation (and scale) parameters *of the camera*,
        # not of the face. In other words, they assume the camera is is translated
        # w.r.t. the model, not the other way around (but it is technically equivalent).
        # Because we assume to have a fixed camera and a moving face, we actually apply
        # translation (and scale) to the model, not the camera.
        T = torch.eye(4, 4, device=self.device).repeat(b, 1, 1)
        T[:, 0, 3] = cam[:, 1]  # translation in X
        T[:, 1, 3] = cam[:, 2]  # translation in Y

        # The same issue applies to the 'scale' parameter
        # which we'll apply to the model, too
        sc = cam[:, 0, None, None]
        S = torch.eye(4, 4, device=self.device).repeat(b, 1, 1) * sc
        S[:, 3, 3] = 1

        if crop_mat is None:

            # Setting crop matrix to identity matrix
            crop_mat = torch.eye(3, device=self.device).repeat(b, 1, 1)

            if not self._warned_about_crop_mat:
                LOGGER.warning(
                    "Arg `crop_mat` is not given, so cannot render in the "
                    "original image space, only in cropped image space!"
                )
                self._warned_about_crop_mat = True

        # Now we have to do something funky. EMOCA/DECA works on cropped images. This is a problem when
        # we want to quantify motion across frames of a video because a face might move a lot (e.g.,
        # sideways) between frames, but this motion is kind of 'negated' by the crop (which will
        # just yield a cropped image with a face in the middle). Fortunately, the smart people
        # at the MPI encoded the cropping operation as a matrix operation (using a 3x3 similarity
        # transform matrix). So what we'll do (and I can't believe this actually works) is to
        # map the vertices all the way from world space to raster space (in which the crop transform
        # was estimated), then apply the inverse of the crop matrix, and then map it back to world
        # space. To do this, we also need a orthographic projection matrix (OP), which maps from
        # world to NDC space, and a viewport matrix (VP), which maps from NDC to raster space.
        # Note that we need this twice: one for the 'forward' transform (world -> crop raster space)
        # and one for the 'backward' transform (full image raster space -> world)

        crop_size = (224, 224)  # fixed by DECA-based models
        OP = create_ortho_matrix(*crop_size, device=self.device)  # forward (world -> cropped NDC)
        VP = create_viewport_matrix(*crop_size, device=self.device)  # forward (cropped NDC -> cropped raster)
        CP = crop_matrix_to_3d(crop_mat)  # crop matrix
        VP_ = create_viewport_matrix(*self.orig_img_size, device=self.device)  # backward (full NDC -> full raster)
        OP_ = create_ortho_matrix(*self.orig_img_size, device=self.device)  # backward (full NDC -> world)

        # Let's define the *full* transformation chain into a single 4x4 matrix
        # (Order of transformations is from right to left)
        pose = S @ T
        forward = torch.inverse(CP) @ VP @ OP
        backward = torch.inverse((VP_ @ OP_))
        mat = backward @ forward @ pose

        v = transform_points(mat, v)
        out = {"v": v}

        # To complete the full transformation matrix, we need to also
        # add the rotation (which was already applied to the data by the
        # FLAME model)
        mat = mat @ R
        out['mat'] = mat

        if self.extract_tex:
            # Note to self: this makes reconstruction very slow!
            out['tex'] = self.D_flame_tex(enc_dict['tex'])

        return out

    def _upsample_mesh(self, v, normals, disp_map):
        """Upsamples a 'coarse' mesh to a 'dense' mesh given a displacement map;
        a pytorch-based implementation of the original code by the MPI (which used numpy).

        Parameters
        ----------
        v : torch.Tensor
            A tensor of shape (B, V, 3) containing the vertices of the mesh
        normals : torch.Tensor
            A tensor of shape (B, V, 3) containing the normals of the mesh
        disp_map : torch.Tensor
            A tensor of shape (B, H, W) containing the displacement map

        Returns
        -------
        v_dense : torch.Tensor
            A tensor of shape (B, Vd, 3) containing the vertices of the dense mesh
        """

        dense_template = self._template['dense']
        x_coords = dense_template["x_coords"]
        y_coords = dense_template["y_coords"]
        valid_pix_idx = dense_template["valid_pix_idx"]
        valid_pix_tris = dense_template["valid_pix_tris"]
        valid_pix_b_coords = dense_template["valid_pix_b_coords"]

        pixel_3d_points = (
            v[:, valid_pix_tris[:, 0], :] * valid_pix_b_coords[:, 0, None]
            + v[:, valid_pix_tris[:, 1], :] * valid_pix_b_coords[:, 1, None]
            + v[:, valid_pix_tris[:, 2], :] * valid_pix_b_coords[:, 2, None]
        )

        pixel_3d_normals = (
            normals[:, valid_pix_tris[:, 0], :]
            * valid_pix_b_coords[:, 0, None]
            + normals[:, valid_pix_tris[:, 1], :]
            * valid_pix_b_coords[:, 1, None]
            + normals[:, valid_pix_tris[:, 2], :]
            * valid_pix_b_coords[:, 2, None]
        )

        pixel_3d_normals /= torch.linalg.norm(pixel_3d_normals, axis=-1, keepdims=True)
        displacements = disp_map[:, y_coords[valid_pix_idx].long(), x_coords[valid_pix_idx].long()]
        offsets = displacements[:, :, None] * pixel_3d_normals
        v_dense = pixel_3d_points + offsets

        return v_dense

    def forward(self, imgs, crop_mat=None):
        """Performs reconstruction of the face as a list of landmarks
        (vertices).

        Parameters
        ----------
        images : torch.Tensor
            A 4D (batch_size x 3 x 224 x 224) ``torch.Tensor`` representing batch of
            RGB images cropped to 224 (h) x 224 (w)

        Returns
        -------
        dec_dict : dict
            A dictionary with two keys: ``"v"``, a numpy array with reconstructed vertices
            (5023 for  'coarse' models or 59315 for 'dense' models) and ``"mat"``, a
            4x4 numpy array representing the local-to-world matrix

        Notes
        -----
        Before calling ``__call__``, you *must* set the ``crop_mat`` attribute to the
        estimated cropping matrix if you want to be able to render the reconstruction
        in the original image (see example below)

        Examples
        --------
        To reconstruct an example, call the ``EMOCA`` object, but make sure to set the
        ``crop_mat`` attribute first:

        >>> from medusa.data import get_example_image
        >>> from medusa.crop import FanCropModel
        >>> img = get_example_image()
        >>> crop_model = FanCropModel(device='cpu')
        >>> cropped_img, crop_mat = crop_model(img)
        >>> recon_model = DecaReconModel(name='emoca-coarse', device='cpu')
        >>> recon_model.crop_mat = crop_mat
        >>> out = recon_model(cropped_img)
        >>> out['v'].shape
        (1, 5023, 3)
        >>> out['mat'].shape
        (1, 4, 4)
        """

        out = {}
        if imgs.shape[2:] != (224, 224):
            # Extract width (imgs.shape[3]) and height (img.shape[2])
            if self.orig_img_size is None:
                self.orig_img_size = imgs.shape[2:][::-1]

            crop_results = self._crop_model(imgs)
            crop_mat = crop_results.pop("crop_mat")
            out["img_idx"] = crop_results.pop("img_idx")
            imgs = crop_results.pop("imgs_crop")
        else:
            # Must be already cropped!
            if self.orig_img_size is None or crop_mat is None:
                # If we don't know the original image size or we don't have access to
                # the crop matrix, we won't be able to render in original space so we'll
                # set the original image size to be the cropped image size
                self.orig_img_size = (224, 224)
                LOGGER.warning("Original image size unkown, rendering in cropped space!")

        imgs = self._preprocess(imgs)
        enc_dict = self._encode(imgs)
        dec_dict = self._decode(enc_dict, crop_mat)
        out = {**out, **dec_dict}

        return out
