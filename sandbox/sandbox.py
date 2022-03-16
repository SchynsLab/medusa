import os
import cv2
import warnings
import torch
import shutil
import numpy as np
import os.path as op
from tqdm import tqdm
from glob import glob

from deca import DECA
from deca.datasets.datasets import TestData 
from deca.utils.config import cfg as deca_cfg

warnings.filterwarnings(
    action='ignore',
    category=UserWarning
)

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning
)

# Initialize DECA reconstruction model
deca_cfg.model.use_tex = False
deca = DECA(config=deca_cfg, device='cuda')

detailed = True
decode_params = dict(
    rendering=True if detailed else False,
    vis_lmk=False,
    return_vis=True if detailed else False,
    use_detail=True if detailed else False
)

preproc = TestData('sandbox/test.jpg', face_detector='fan', device='cuda')[0]
    
# Store tensor on device and use batch size of 1 (otherwise error)
img_tensor = preproc['image'].to('cuda')[None, ...]

with torch.no_grad():  # start recon!
    codedict = deca.encode(img_tensor)
    out = deca.decode(codedict, **decode_params)
    if decode_params['return_vis']:
        dec_dict, vis_dict = out
    else:
        dec_dict = out
    
    np.save('pyface/data/example_obj.npy', dec_dict['verts'].to('cpu').squeeze())
    deca.save_obj('pyface/data/example_obj.obj', dec_dict)