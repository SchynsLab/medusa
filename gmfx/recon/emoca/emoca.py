from trimesh import transform_points
import yaml
import torch
from pathlib import Path

from .models.encoders import ResnetEncoder
from .models.decoders import FLAME, FLAMETex, Generator
from .lbs import batch_rodrigues, batch_rot_matrix_to_ht, batch_orth_proj_matrix

# May have some speed benefits
torch.backends.cudnn.benchmark = True


class EMOCA(torch.nn.Module):
    def __init__(self, cfg=None, device='cuda'):
        """ Initializes an EMOCA model object.
        
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
        self._load_cfg(cfg)  # sets self.cfg
        self._image_size = self.cfg['DECA']['image_size']
        self._create_submodels()
        
    def _load_cfg(self, cfg):
        """ Loads a (default) config file. """
        if cfg is None:
            cfg = self.package_root / 'configs/emoca.yaml'

        with open(cfg, 'r') as f_in:
            self.cfg = yaml.safe_load(f_in)
            
        # Make sure the paths are absolute!
        for section, options in self.cfg.items():
            for key, value in options.items():
                if 'path' in key:
                    # We assume data is stored at one dire above the package root
                    self.cfg[section][key] = str(self.package_root.parent / value)

    def _create_submodels(self):
        """ Creates all EMOCA encoding and decoding submodels. To summarize:
            - `E_flame`: predicts (coarse) FLAME parameters given an image
            - `E_expression`: predicts expression FLAME parameters given an image
            - `E_detail`: predicts detail FLAME parameters given an image
            - `D_flame`: outputs a ("coarse") mesh given (shape, exp, pose) FLAME parameters
            - `D_flame_tex`: outputs a texture map given (tex) FLAME parameters
            - `D_detail`: outputs detail map (in uv space) given (detail) FLAME parameters
        """
        # set up parameter list and dict
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

        # Load weights from checkpoint and apply to models
        model_path = self.cfg['DECA']['model_path']
        checkpoint = torch.load(model_path)
        
        self.E_flame.load_state_dict(checkpoint['E_flame'])
        self.E_detail.load_state_dict(checkpoint['E_detail'])
        self.E_expression.load_state_dict(checkpoint['E_expression'])
        self.D_detail.load_state_dict(checkpoint['D_detail'])        

        self.E_expression.cuda()  # for some reason E_exp should be explicitly cast to cuda

        # Set everything to 'eval' (inference) mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.E_expression.eval()
        self.D_detail.eval()
        torch.set_grad_enabled(False)  # apparently speeds up forward pass, too

    def encode(self, image):
        """ "Encodes" the image into FLAME parameters, i.e., predict FLAME
        parameters for the given image. Note that, at the moment, it only
        works for a single image, not a batch of images.
        
        Parameters
        ----------
        image : torch.Tensor
            A Tensor with shape 1 (batch size) x 3 (color ch.) x 244 (w) x 244 (h)
        
        Returns
        -------
        enc_dict : dict
            A dictionary with all encoded parameters and some extra data needed
            for the decoding stage.
        """

        # Encode image into FLAME parameters, then decompose parameters
        # into a dict with parameter names (shape, tex, exp, etc) as keys
        # and the estimated parameters as values
        enc_params = self.E_flame(image)
        enc_dict = self._decompose_params(enc_params, self.param_dict)

        # Note to self:
        # enc_dict['cam'] contains [batch_size, x_trans, y_trans, zoom] (in mm?)
        # enc_dict['pose'] contains [rot_x, rot_y, rot_z] (in radians) for the neck
        # and jaw (first 3 are neck, last three are jaw)
        # rot_x_jaw = mouth opening
        # rot_y_jaw = lower jaw to left or right
        # rot_z_jaw = not really possible?
        
        # Encode image into detail parameters
        #detail_params = self.E_detail(image)
        #enc_dict['detail'] = detail_params

        # Replace "DECA" expression parameters with EMOCA-specific
        # expression parameters       
        enc_dict['exp'] = self.E_expression(image)

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
                # Reshape 27 flattened params into 9 x 3 array
                enc_dict[key] = enc_dict[key].reshape(enc_dict[key].shape[0], 9, 3)

        return enc_dict

    def decode(self, enc_dict):
        """ Decodes the face attributes (vertices, landmarks, texture, detail map)
        from the encoded parameters.
        
        Parameters
        ----------
        tform : torch.Tensor
            A Torch tensor containing the similarity transform parameters
            that map points on the cropped image to the uncropped image; this 
            is availabe from the `tform_params` attribute of the initialized
            FAN object (e.g., `fan.tform_params`); if `None`, the vertices
            will not be transformed to the original image space
        orig_size : tuple
            Tuple containing the original image size (height, width), i.e., 
            before cropping; needed to transform and render the mesh in the
            original image space

        Returns
        -------
        dec_dict : dict
            A dictionary with the results from the decoding stage

        Raises
        ------
        ValueError
            If `tform` parameter is not `None` and `orig_size` is `None`. In other
            words, if `tform` is supplied, `orig_size` should be supplied as well
            
        """
        
        # "Decode" vertices (V), 2D landmarks (lm2d), and 3D landmarks (lm3d)
        # given shape, expression (exp), and pose FLAME parameters
        # Note that V is in world space (i.e., no translation/scale applied yet)
        v, R, _, _ = self.D_flame(shape_params=enc_dict['shape'],
                                  expression_params=enc_dict['exp'],
                                  pose_params=enc_dict['pose'])

        cam = enc_dict['cam'].squeeze()

        # # Add translation and scale
        v[:, :, 0] = v[:, :, 0] + cam[1]
        v[:, :, 1] = v[:, :, 1] + cam[2]
        v = v * cam[0]

        R = torch.mean(R[0, :, :, :], dim=0)
        T = torch.eye(4).to(R.device)
        T[0, 3] = cam[1]
        T[1, 3] = cam[2]
        S = torch.eye(4).to(R.device) * cam[0]
        S[3, 3] = 1

        mat = (S @ T) @ R
    
        # v = torch.squeeze(v)
        # v = torch.column_stack((v, torch.ones(v.shape[0]).to('cuda')))
        # v = torch.matmul(v, torch.transpose(torch.linalg.inv(mat), 0, 1))
        # v = v[:, :3].unsqueeze(0)

        # v = v * 10

        #tex = self.D_flame_tex(enc_dict['tex'])
        return {'v': v.cpu().numpy().squeeze(), 'mat': mat.cpu().numpy()}
        
        # # Decode detail map (uses jaw rotations, i.e., pose[3:] as well as exp and of course detail parameters)
        # #inp_D_detail = torch.cat([enc_dict['pose'][:, 3:], enc_dict['exp'], enc_dict['detail']], dim=1)
        # #uv_z = self.D_detail(inp_D_detail)
            
    def forward(self, img):
        enc_dict = self.encode(img)
        dec_dict = self.decode(enc_dict)
        return dec_dict