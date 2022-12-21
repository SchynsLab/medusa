def main():
    # avoid cyclical import
    from . import get_example_video
    from ..recon import videorecon

    vid = get_example_video()
    for model in ["mediapipe", "emoca-coarse"]:
        data = videorecon(vid, recon_model=model)
        data.save(str(vid).replace(".mp4", f"_{model}.h5"))
