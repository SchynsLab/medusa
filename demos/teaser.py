import cv2
import numpy as np
from pathlib import Path
from skimage.transform import rescale
from medusa.data import load_h5 
from medusa.render import Renderer
from medusa.preproc import videorecon
from imageio import get_writer, get_reader
from pygifsicle import optimize

SCALE = 0.2
VID = 'pexels_2.mp4'

h5_out = Path(f'demos/pexels_2_desc-recon_shape.h5')
if not h5_out.is_file():
    data = videorecon('test_data/pexels/pexels_2.mp4', recon_model_name="mediapipe",
                      out_dir='demos/', render_recon=False)
else:
    data = load_h5(h5_out)


fps = data.sf
w, h = data.img_size
w, h = int(w * SCALE), int(h * SCALE)

renderer = Renderer(camera_type='intrinsic', wireframe=True, cam_mat=data.cam_mat,
                    viewport=(w, h))

font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (255, 255, 255)
writer = get_writer('demos/teaser.gif', mode='I', fps=fps)

### INTRODUCTION ###
# txt1_pos = (int(650 * SCALE), int(500 * SCALE))
# txt2_pos = (int(250 * SCALE), int(620 * SCALE))
# for i in range(70):
#     img = np.zeros((h, w, 4)).astype(np.uint8)
#     cv2.putText(img, 'Medusa:', txt1_pos, font, 5 * SCALE, font_color,
#                 1, cv2.LINE_AA)
#     cv2.putText(img, '4D face reconstruction and analysis', txt2_pos,
#                 font, 2.5 * SCALE, font_color, 1, cv2.LINE_AA)
    
#     writer.append_data(img)
    
#     if i > 50:
#         font_color = [fc - 20 for fc in font_color]

reader = get_reader('test_data/pexels/pexels_2.mp4')
for i, img in enumerate(reader):
    img = rescale(img, SCALE, anti_aliasing=True, channel_axis=2,
                  preserve_range=True).astype(np.uint8)
    alpha = np.min([i / 100, 1]) #np.zeros(img.shape[:2]) + i
    img = (img * alpha).astype(np.uint8)
    cv2.putText(img, 'Medusa reconstructs 3D face shape from videos',
                (int(100 * SCALE), int(100 * SCALE)),
                font, 1.5 * SCALE, (255, 255, 255), 1, cv2.LINE_AA)

    if i > 50:
        recon = renderer(data.v[i, :, :], f=data.f)
        alpha = np.min([(i - 50) / 100, 1])
        img = renderer.alpha_blend(recon, img, face_alpha=alpha)
    
    writer.append_data(img)
    if i > 150:
        break

writer.close()
reader.close()

#optimize('demos/teaser.gif')