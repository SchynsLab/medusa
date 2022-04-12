import cv2
import torch
import warnings
import shutil
import imageio
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from ..detect import FAN
from ..recon import EMOCA
from ..recon.emoca.utils import tensor2image
from ..io import Data


warnings.filterwarnings(
    action='ignore',
    category=UserWarning
)

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning
)


def recon(in_dir, out_dir, participant_label, cfg, device):
    """ Reconstruction of all frames of a video. 
    
    Parameters
    ----------
    in_dir : str
        Path to data directory, which should contain subdirectories for
        different participants (e.g., 'sub-01', 'sub-02')
    out_dir : str
        Path to output directory; will be created if it doesn't exist yet
    participant_label : str
        Like in BIDS, this indicates which participant from the `in_dir` will
        be processed
    cfg : str
        Path to config file for reconstruction
    device : str
        Either "cuda" (for GPU) or "cpu"
    """

    data_dir = Path(in_dir) / participant_label
    
    if out_dir is None:
        out_dir = Path(in_dir) / 'derivatives' / participant_label / 'recon'
    
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    # Init here, otherwise it's done every frame -> slow
    fan = FAN(device=device)  # for face detection / cropping
    emoca = EMOCA(cfg=cfg, device=device)

    vids = list(data_dir.glob('*_video.mp4'))
    if not vids:
        raise ValueError(f"Could not find any MP4 videos in {data_dir}!")

    # Find files and loop over them
    for vid in vids:
        base_f = vid.stem
        reader = imageio.get_reader(vid)
        fps = reader.get_meta_data()['fps']

        # Use cv2 to get n_frames (not possible with imageio ...)
        cap = cv2.VideoCapture(str(vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ft_path = Path(str(vid).replace('_video.mp4', '_frametimes.tsv'))
        if ft_path.exists():
            frame_t = pd.read_csv(ft_path, sep='\t')['t'].to_numpy()
        else:
            print(f"Warning: did not find frametimes file for video {vid}!")
            frame_t = None

        ev_path = Path(str(vid).replace('_video.mp4', '_events.tsv'))
        if ev_path.exists():
            events = pd.read_csv(ev_path, sep='\t')
        else:
            print(f"Warning: did not find events file for video {vid}!")
            events = None

        # Init lists for reconstruced vertices across time (v_4D) and
        # the framewise motion estimates (motion; global rot x/y/z, trans x/y, scale (z))
        v_4D = []
        motion = []

        # For QC, we will write out the rendered shape on top of the original frame
        f_out = str(out_dir / base_f) + '_desc-recon'
        writer = imageio.get_writer(f_out + '_viz-orig_shape.mp4', mode='I', fps=fps)
        
        # Loop across frames of video
        for i, frame in tqdm(enumerate(reader), desc=f'Recon {base_f}', total=n_frames):

            # Prepare frame to be used as background during rendering            
            orig_size = frame.shape[:2]
            frame_orig = (frame / 255.).transpose(2, 0, 1)  # channel, width, height
            frame_orig = torch.tensor(frame_orig).float()[None, ...].to('cuda')
    
            # Crop image and do forward pass on encoding stage
            frame_cropped = fan(frame)
            enc_dict = emoca.encode(frame_cropped)

            # Decode with `tform` params, so that the reconstructed vertices are in "image space"
            # which are subsequently rendered and passed to the writer; note that these operations
            # are for QC only and can thus be technically omitted
            dec_dict = emoca.decode(enc_dict, tform=fan.tform_params, orig_size=orig_size)
            rend_dict = emoca.render_dec(enc_dict, dec_dict, render_world=False,
                                        img_orig=frame_orig)

            img = tensor2image(rend_dict['shape_detail'][0], bgr=False)
            writer.append_data(img)
            
            # First, save motion parameters from the encoding stage
            motion.append(torch.cat((enc_dict['pose'][:, :3], enc_dict['cam']), axis=1))
            enc_dict['pose'][0, :3] = 0
            
            # Now, we decode again, but this time, we'll set the first three
            # pose params to 0 to get the reconstruction without any
            # rotation (also `tform` is set to `None` so everything is in
            # "world space")
            dec_dict = emoca.decode(enc_dict, tform=None)
            
            # We'll only save the `V` vertices, which are in "world space"
            # and thus not translated / scaled (so they do not contain
            # any global/rigid motion)
            v_4D.append(dec_dict['V'])
            if i > 20:
                break
                 
        reader.close()
        writer.close()

        # Stack v_4D into 3D arr and motion into 2D arr
        v_4D = torch.cat(v_4D)
        motion = torch.cat(motion).cpu().numpy()

        # Create Data object and save to disk as hdf5 file
        data = Data(
            v=v_4D.cpu().numpy(), motion=motion,
            frame_t=frame_t, events=events, fps=fps, dense=False
        )
        data.save(f_out + '_shape.h5')

        # Also create video of "world space" reconstruction
        writer = imageio.get_writer(f_out + '_viz-world_shape.mp4', mode='I', fps=fps)
        for i in tqdm(range(data.v.shape[0]), desc=f'Render {base_f}'):
            V = torch.unsqueeze(v_4D[i, ...], 0)  # add singleton dim, needed for render_shape
            V_trans = V.clone() * 7  # zoom
            V_trans[..., 1:] = -V_trans[..., 1:]
            res = emoca.render.render_shape(V, V_trans, h=224, w=224, images=None)
            writer.append_data(tensor2image(res[0], bgr=False))

        writer.close()
        
        # Plot motion
        # FIXME: separate subplot for rot and trans/scale
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(motion)
        fig.savefig(f_out + '_motion.png')
        plt.close()