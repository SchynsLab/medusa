import cv2
import torch
import numpy as np
from ..models.deca.datasets import VideoFrame 


def recon_deca(frame, deca, fan, device='cuda', dense=False, no_crop=False):

    decode_params = dict(
        rendering=True if dense else False,
        vis_lmk=False,
        return_vis=True if dense else False,
        use_detail=True if dense else False
    )

    preproc = VideoFrame(frame, fan, iscrop=False if no_crop else True)[0]
    img = (preproc['image'][0].cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Store tensor on device and use batch size of 1 (otherwise error)
    img_tensor = preproc['image'].to(device)[None, ...]
    with torch.no_grad():  # start recon!
        codedict = deca.encode(img_tensor)
        out = deca.decode(codedict, **decode_params)
        if decode_params['return_vis']:
            dec_dict, vis_dict = out
        else:
            dec_dict = out
        
        # Convert from tensor to numpy
        if dense:
            v = deca.get_dense_verts(dec_dict)
        else:
            v = dec_dict['trans_verts'][0].cpu().numpy()

    return v, img