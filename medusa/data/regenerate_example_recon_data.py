"""Regenerates example reconstruction data."""

def main():
    """Utility cli command (see ``cli.py``) to regenerate the example data for when
    the Data4D class changes."""
    # avoid cyclical import
    from ..recon import videorecon
    from . import get_example_video

    vid = get_example_video()
    for model in ["mediapipe", "emoca-coarse"]:
        data = videorecon(vid, recon_model=model, loglevel='WARNING')
        data.save(str(vid).replace(".mp4", f"_{model}.h5"))
