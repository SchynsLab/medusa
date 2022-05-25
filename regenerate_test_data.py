from medusa.preproc import videorecon
from medusa.data import get_example_video

vid = get_example_video(as_path=False)
for model in ['mediapipe', 'fan', 'emoca']:
    data = videorecon(vid, recon_model_name=model)
    data.save(vid.replace('.mp4', f'_desc-recon_model-{model}_shape.h5'))