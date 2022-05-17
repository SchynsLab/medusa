from pathlib import Path
from ..core import load_h5


def get_example_video(as_path=True):
    """Retrieves the path to an example video file.

    Parameters
    ----------
    as_path : bool
        Returns the path as a ``pathlib.Path`` object

    Returns
    -------
    path : str, pathlib.Path
        A string or Path object pointing towards the example
        video file

    Examples
    --------
    >>> path = get_example_video(as_path=True)
    >>> path.is_file()
    True
    """

    here = Path(__file__).parent
    path = here / "example_vid.mp4"

    if not as_path:
        path = str(path)

    return path


def get_example_h5(load=False, model="mediapipe", as_path=True):
    """Retrieves an example hdf5 file with reconstructed 4D
    data from the example video.

    Parameters
    ----------
    load : bool
        Whether to return the hdf5 file loaded in memory (``True``)
        or to just return the path to the file
    model : str
        Model used to reconstruct the data; either 'mediapipe' or
        'emoca'
    as_path : bool
        Whether to return the path as a ``pathlib.Path`` object (``True``)
        or just a string (``False``); ignored when ``load`` is ``True``

    Returns
    -------
    MediapipeData, FlameData, str, Path
        When ``load`` is ``True``, returns either a ``MediapipeData``
        or a ``FlameData`` object, otherwise a string or ``pathlib.Path``
        object to the file

    Examples
    --------
    >>> path = get_example_h5(load=False, as_path=True)
    >>> path.is_file()
    True

    # Get hdf5 file already loaded in memory
    >>> data = get_example_h5(load=True, model='mediapipe')
    >>> data.recon_model_name
    'mediapipe'
    >>> data.v.shape  # check out reconstructed vertices
    (232, 468, 3)
    """
    here = Path(__file__).parent
    path = here / f"example_vid_desc-recon_model-{model}_shape.h5"

    if not as_path:
        path = str(path)

    if load:
        return load_h5(path)
    else:
        return path
