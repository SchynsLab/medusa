from pathlib import Path

from skimage import io


def get_example_img(load=False):
    """Loads a test image from the ``flame/data`` directory (an image generated
    on https://this-person-does-not-exist.com).

    Parameters
    ----------
    load : bool
        Whether to load the image into memory as a numpy array or just return the
        filepath

    Returns
    img : pathlib.Path, np.ndarray
        Image or path to image, depending on the ``load`` parameter
    """

    img = Path(__file__).parent / 'example_img.jpg'
    if load:
        img = io.imread(img)

    return img
