from medusa.recon import videorecon
from medusa.data import get_example_video

vid = get_example_video()
for model in ['mediapipe', 'emoca-coarse']:
    data = videorecon(vid, recon_model=model)
    data.save(vid.replace('.mp4', f'_{model}.h5'))
