from medusa.recon import videorecon
from medusa.data import get_example_video

vid = get_example_video(as_path=False)
for model in ['mediapipe', 'fan', 'emoca-coarse']:
    data = videorecon(vid, recon_model=model)
    data.save(vid.replace('.mp4', f'_{model}.h5'))
