"""Regenerates example reconstruction data."""

def main():
    """Utility cli command (see ``cli.py``) to regenerate the example data for when
    the Data4D class changes."""
    # avoid cyclical import
    from ..recon import videorecon
    from . import get_example_video

    for f in [None, 1, 2, 3, 4]:
        vid = get_example_video(n_faces=f)

        for model in ["mediapipe", "emoca-coarse"]:
            data = videorecon(vid, recon_model=model, loglevel='INFO')
            f_out = vid.parents[1] / 'recons' / (vid.stem + f'_{model}.h5')
            data.save(f_out)
