from pathlib import Path

import h5py


def get_template_flame(dense=False):
    """Returns the template (vertices and triangles) of the canonical Flame
    model, in either its dense or coarse version.

    Parameters
    ----------
    dense : bool
        Whether to load in the dense version of the template (``True``) or the coarse
        version (``False``)

    Returns
    -------
    template : dict
        Dictionary with vertices ("v") and faces ("f")

    Examples
    --------
    Get the vertices and faces (triangles) of the standard Flame topology (template) in
    either the coarse version (``dense=False``) or dense version (``dense=True``)

    >>> template = get_template_flame(dense=False)
    >>> template['v'].shape
    (5023, 3)
    >>> template['f'].shape
    (9976, 3)
    >>> template = get_template_flame(dense=True)
    >>> template['v'].shape
    (59315, 3)
    >>> template['f'].shape
    (117380, 3)
    """

    file = Path(__file__).parent / "flame_template.h5"
    with h5py.File(file, "r") as data:

        template_h5 = data["dense" if dense else "coarse"]
        template = {"v": template_h5["v"][:], "f": template_h5["f"][:]}

    return template


if __name__ == "__main__":
    get_template_flame()
