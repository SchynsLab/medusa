import yaml
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from skimage.io import imread

from .models.encoders import ResnetEncoder
from .models.decoders import FLAME, FLAMETex, Generator
from .utils import vertex_normals, tensor_vis_landmarks
from ...render.renderer import SRenderY

# May have some speed benefits
torch.backends.cudnn.benchmark = True


class DECA(torch.nn.Module):
    def __init__(self, cfg=None, device='cuda'):
        """ Initializes a DECA object.
        
        Parameters
        ----------
        cfg : str
            Path to YAML config file. If `None` (default), it
            will use the package's default config file.
        device : str
            Either 'cuda' (uses GPU) or 'cpu'
        """
        super().__init__()
        self.device = device
        self.package_root = Path(__file__).parents[2].resolve()
        self._load_cfg(cfg)
        self.image_size = self.cfg['DECA']['image_size']
        self.uv_size = self.cfg['DECA']['uv_size']
        self._create_submodels()
        self._setup_renderer()

    def _load_cfg(self, cfg):
        
        if cfg is None:
            cfg = self.package_root / 'configs/emoca.yaml'

        with open(cfg, 'r') as f_in:
            self.cfg = yaml.safe_load(f_in)
            
        # Make sure the paths are absolute!
        for section, options in self.cfg.items():
            for key, value in options.items():
                if 'path' in key:
                    self.cfg[section][key] = str(self.package_root.parent / value)

    def _setup_renderer(self):
        
        self.render = SRenderY(self.image_size, obj_filename=self.cfg['DECA']['topology_path'],
                               uv_size=self.uv_size).to(self.device)

        # Load required data
        mask = imread(self.cfg['DECA']['face_eye_mask_path']).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.uv_size, self.uv_size]).to(self.device)
        
        mask = imread(self.cfg['DECA']['face_mask_path']).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.uv_size, self.uv_size]).to(self.device)
        
        # displacement correction
        fixed_dis = np.load(self.cfg['DECA']['fixed_displacement_path'])
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)

    def _create_submodels(self):
        # set up parameters
        self.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        self.num_list = [self.cfg['E_flame'][f'n_{param}'] for param in self.param_list]
        self.param_dict = {self.param_list[i]: self.num_list[i] for i in range(len(self.param_list))}
        self.n_param = sum(self.num_list)
        
        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.E_expression = ResnetEncoder(self.cfg['E_flame']['n_exp'])
        self.E_detail = ResnetEncoder(outsize=self.cfg['E_detail']['n_detail']).to(self.device)

        # decoders
        self.D_flame = FLAME(self.cfg['D_flame'], self.cfg['E_flame']['n_shape'],
                             self.cfg['E_flame']['n_exp']).to(self.device)

        if self.cfg['DECA']['use_tex']:
            self.D_flame_tex = FLAMETex(self.cfg['D_flame'],
                                        self.cfg['E_flame']['n_tex']).to(self.device)

        latent_dim = self.cfg['E_detail']['n_detail'] + self.cfg['E_flame']['n_exp'] + 3
        self.D_detail = Generator(latent_dim=latent_dim, out_channels=1,
                                  out_scale=self.cfg['D_detail']['max_z'], sample_mode='bilinear').to(self.device)

        # Load weights
        model_path = self.cfg['DECA']['model_path']

        checkpoint = torch.load(model_path)
        self.E_flame.load_state_dict(checkpoint['E_flame'])
        self.E_detail.load_state_dict(checkpoint['E_detail'])
        self.E_expression.load_state_dict(checkpoint['E_expression'])
        self.D_detail.load_state_dict(checkpoint['D_detail'])        

        self.E_expression.cuda()  # for some reason E_exp should be explicitly cast to cuda

        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.E_expression.eval()
        self.D_detail.eval()
        torch.set_grad_enabled(False)

    def encode(self, images):
        
        params = self.E_flame(images)
        enc_dict = self._decompose_params(params, self.param_dict)
        enc_dict['img_crop'] = images  # store images, too
    
        # Note to self:
        # enc_dict['cam'] contains [batch_size, x_trans, y_trans, zoom] (in mm?)
        # enc_dict['pose'] contains [rot_x, rot_y, rot_z] (in radians) for the neck
        # and jaw (first 3 are neck, last three are jaw)
        # rot_x_jaw = mouth opening
        # rot_y_jaw = lower jaw to left or right
        # rot_z_jaw = not really possible?
        
        # Get detail parameters
        detail_params = self.E_detail(images)
        enc_dict['detail'] = detail_params

        # Replace DECA exp parameters with EMOCA        
        enc_dict['exp'] = self.E_expression(images)
        
        return enc_dict

    def _decompose_params(self, parameters, num_dict):
        """ Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']. """
        enc_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            enc_dict[key] = parameters[:, start:end]
            start = end
            if key == 'light':
                enc_dict[key] = enc_dict[key].reshape(enc_dict[key].shape[0], 9, 3)

        return enc_dict

    def decode(self, enc_dict, tform=None, orig_size=None):
        """ Decodes the face attributes (vertices, landmarks, texture, detail map)
        from the encoded parameters.
        
        Parameters
        ----------
        tform : torch.Tensor
            A Torch tensor containing the similarity transform parameters
            that map points on the cropped image to the uncropped image;
            if `None`, the vertices will not be transformed to the original
            image space
        orig_size : tuple
            Tuple containing the original image size (height, width)
            
        """
        dec_dict = {}
        V, lm2d, lm3d = self.D_flame(shape_params=enc_dict['shape'],
                                     expression_params=enc_dict['exp'],
                                     pose_params=enc_dict['pose'])

        if self.cfg['DECA']['use_tex']:
            tex = self.D_flame_tex(enc_dict['tex'])
        else:
            tex = torch.zeros([V.shape[0], 3, self.uv_size, self.uv_size], device=self.device) 

        # Orthographic projection of vertices and landmarks
        V_trans = self.batch_orth_proj(V, enc_dict['cam'])
        V_trans[:, :, 1:] = -V_trans[:, :, 1:]

        lm2d_trans = self.batch_orth_proj(lm2d, enc_dict['cam'])[:, :, :2]
        lm2d_trans[:, :, 1:] = -lm2d_trans[:, :, 1:]
        lm3d_trans = self.batch_orth_proj(lm3d, enc_dict['cam'])
        lm3d_trans[:, :, 1:] = -lm3d_trans[:, :, 1:]

        if tform is not None:
            # Transform to non-cropped image space
            points_scale = [self.image_size, self.image_size]

            if orig_size is None:
                raise ValueError("If `tform` is not None, the `orig_size` argument "
                                 "should also be set!")
            
            # Transform to original image space
            V_trans = transform_points(V_trans, tform, points_scale, orig_size)
            lm2d_trans = transform_points(lm2d_trans, tform, points_scale, orig_size)
            lm3d_trans = transform_points(lm3d_trans, tform, points_scale, orig_size)

        # Decode detail
        inp_D_detail = torch.cat([enc_dict['pose'][:, 3:], enc_dict['exp'], enc_dict['detail']], dim=1)
        uv_z = self.D_detail(inp_D_detail)

        dec_dict['V'] = V
        dec_dict['V_trans'] = V_trans
        dec_dict['lm2d'] = lm2d
        dec_dict['lm2d_trans'] = lm2d_trans
        dec_dict['lm3d'] = lm3d        
        dec_dict['lm3d_trans'] = lm3d_trans        
        dec_dict['tex'] = tex
        dec_dict['detail'] = uv_z

        return dec_dict

    def render_dec(self, enc_dict, dec_dict, render_world=False, img_orig=None, zoom=10):
        """ Render images of 3D reconstruction.
        
        enc_dict : dict
            Dictionary with encoding parameters
        dec_dict : dict
            Dictionary with decoding results
        """
        
        V, V_trans, tex = dec_dict['V'], dec_dict['V_trans'], dec_dict['tex'] 
        if render_world:
            V_trans = V.clone()
            V_trans = V_trans * zoom
            V_trans[:, :, 1:] = -V_trans[:, :, 1:]

        # images (shape+tex), albedo_images, alpha_images (visibility?)
        # pos_mask (visibility?), shading_images, grid (224, 224, 2)
        # normals (5023, 3), normal_images (3, 224, 224), 
        # transformed_normals (5023, 3)
        render_dict = self.render(V, V_trans, tex, enc_dict['light'])

        if render_world:
            # Render in world space
            h, w = None, None
            background = None
        elif img_orig is None:
            # Render on top of cropped image
            h, w = enc_dict['img_crop'].shape[2:]
            background = enc_dict['img_crop']
        else:
            # Render on top of original image
            h, w = img_orig.shape[2:]
            background = img_orig

        shape_img, _, grid, alpha_img = self.render.render_shape(V, V_trans, h=h, w=w, images=background, return_grid=True)
        render_dict['shape'] = shape_img
        render_dict['alpha'] = alpha_img

        if not render_world:  # FIXME
            render_dict['lm2d'] = tensor_vis_landmarks(background, dec_dict['lm2d_trans'])
            render_dict['lm3d'] = tensor_vis_landmarks(background, dec_dict['lm3d_trans'])

        uv_detail_normals = self.displacement2normal(dec_dict['detail'], V, render_dict['normals'])
        uv_shading = self.render.add_SHlight(uv_detail_normals, enc_dict['light'])
        uv_texture = tex * uv_shading

        render_dict['uv_texture'] = uv_texture 
        render_dict['uv_detail_normals'] = uv_detail_normals
        render_dict['displacement_map'] = dec_dict['detail'] + self.fixed_uv_dis[None, None, :, :]
    
        dni = F.grid_sample(uv_detail_normals, grid, align_corners=False) * alpha_img
        render_dict['detail_normals'] = dni
        render_dict['shape_detail'] = self.render.render_shape(V, V_trans, detail_normal_images=dni, h=h, w=w, images=background)
        
        return render_dict

    def batch_orth_proj(self, X, cam):
        ''' orthgraphic projection
            X:  3d vertices, [bz, n_point, 3]
            camera: scale and translation, [bz, 3], [scale, tx, ty]
        '''
        cam = cam.clone().view(-1, 1, 3)
        
        # Add x/y translation and add back z
        X_trans = X[:, :, :2] + cam[:, :, 1:]
        X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)

        # Scale X_trans by scale parameter
        return cam[:, :, 0:1] * X_trans
        
    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals + self.fixed_uv_dis[None, None, :, :] * uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        return uv_detail_normals

    def model_dict(self):
        return {
            'E_flame': self.E_flame.state_dict(),
            'E_detail': self.E_detail.state_dict(),
            'D_detail': self.D_detail.state_dict()
        }


def transform_points(points, tform, points_scale=None, out_scale=None):

    points_2d = points[:, :, :2]
    
    #'input points must use original range'
    if points_scale:
        assert points_scale[0] == points_scale[1]
        points_2d = (points_2d * 0.5 + 0.5) * points_scale[0]

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
        torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1),
        tform
    ) 

    if out_scale: # h,w of output image size
        trans_points_2d[:, :, 0] = trans_points_2d[:, :, 0] / out_scale[1] * 2 - 1
        trans_points_2d[:, :, 1] = trans_points_2d[:, :, 1] / out_scale[0] * 2 - 1
    
    trans_points = torch.cat([trans_points_2d[:, :, :2], points[:, :, 2:]], dim=-1)
    return trans_points
