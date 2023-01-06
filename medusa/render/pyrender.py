"""Module with functionality to render 4D face mesh data.

The ``Renderer`` class is a high-level wrapper around functionality from the
excellent `pyrender <https://pyrender.readthedocs.io>`_ package [1]_.

.. [1] Matl, Matthew. *pyrender* [computer software]. https://github.com/mmatl/pyrender
"""

import warnings

import numpy as np
import torch
from OpenGL.GL import glLineWidth
from pyrender import (DirectionalLight, Mesh, Node, OffscreenRenderer,
                      OrthographicCamera, PerspectiveCamera, Scene)
from pyrender.constants import RenderFlags
from trimesh import Trimesh, visual

from ..defaults import DEVICE
from .base import BaseRenderer


class PyRenderer(BaseRenderer):
    """A high-level wrapper around a pyrender-based renderer.

    Parameters
    ----------
    viewport : tuple[int]
        Desired output image size (width, height), in pixels; should match
        the original image (before cropping) that was reconstructed
    cam_mat : torch.tensor
        A camera matrix to set the position/angle of the camera
    cam_type : str
        Either 'orthographic' (for Flame-based reconstructions) or
        'perpective' (for mediapipe reconstructions)
    shading : str
        Type of shading ('flat', 'smooth', or 'wireframe')
    wireframe_opts : None, dict
        Dictionary with extra options for wireframe rendering (options: 'width', 'color')
    device : str
        Device to store the image on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        viewport,
        cam_mat=None,
        cam_type="orthographic",
        shading="flat",
        wireframe_opts=None,
        device=DEVICE,
    ):
        """Initializes a Pyrenderer object."""
        self.viewport = viewport
        self.cam_mat = cam_mat
        self.cam_type = cam_type
        self.shading = shading
        self.wireframe_opts = wireframe_opts
        self.device = device  # for compatibility only
        self._scene = self._create_scene()
        self._renderer = OffscreenRenderer(*self.viewport)  # actual pyrender renderer
        self._misc_config()

    def __str__(self):
        """Returns the name of the renderer (nice for testing)."""
        return "pyrender"

    def _create_scene(self):
        """Creates a simple scene with a camera and a directional light.

        The object (reconstructed face) is added when calling __call__.
        """
        # TODO: make bg_color an init parameter
        scene = Scene(bg_color=[0, 0, 0, 0], ambient_light=(255, 255, 255))

        if self.cam_type == "orthographic":
            camera = OrthographicCamera(xmag=1, ymag=1, znear=0.01)
        elif self.cam_type == "perspective":
            fov = 2 * np.arctan(self.viewport[1] / (2 * self.viewport[0]))  # in rad!
            camera = PerspectiveCamera(yfov=fov, znear=1.0, zfar=10000.0)
        else:
            raise ValueError(f"Unknown camera type {self.cam_type}")

        if self.cam_mat is None:
            cam_mat = np.eye(4)
        else:
            cam_mat = self.cam_mat
            if torch.is_tensor(cam_mat):
                cam_mat = cam_mat.cpu().numpy()

        camera_node = Node(camera=camera, matrix=cam_mat)
        scene.add_node(camera_node)
        light = DirectionalLight(intensity=5.0)
        scene.add_node(Node(light=light, matrix=cam_mat))

        return scene

    def _misc_config(self):
        """Do some configurations."""
        if self.shading == "wireframe":

            if self.wireframe_opts is None:
                self.wireframe_opts = {}

            width = self.wireframe_opts.get("width", None)
            if width is not None:
                glLineWidth(width)

            color = self.wireframe_opts.get("color", None)
            if color is not None:
                self.wireframe_opts["color"] = np.array(color, dtype=np.float32)
            else:
                self.wireframe_opts["color"] = np.array(
                    [1.0, 0.0, 0.0, 1], dtype=np.float32
                )

    def __call__(self, v, tris, overlay=None):
        """Performs the actual rendering for a given mesh. Note that this
        function does not do proper batched (vectorized) rendering, but is made
        to look like it does to keep it consistent with the
        ``PytorchRenderer``.

        Parameters
        ----------
        v : torch.tensor
            A 3D (batch size x vertices x 3) tensor with vertices
        tris : torch.tensor
            A 3D (batch size x vertices x 3) tensor with triangles
        overlay : torch.tensor
            A tensor with shape (batch size x vertices) with vertex colors

        Returns
        -------
        img : torch.tensor
            A 4D tensor with uint8 values of shape batch size x h x w x 3 (RGB), where
            h and w are defined in the viewport
        """

        v, tris, overlay = self._preprocess(v, tris, overlay, format="numpy")

        for i in range(v.shape[0]):

            if overlay is not None:
                overlay_ = overlay[i]
            else:
                overlay_ = None

            mesh = Trimesh(v[i], tris, vertex_colors=overlay_)
            mesh = Mesh.from_trimesh(
                mesh,
                smooth=self.shading == "smooth",
                wireframe=self.shading == "wireframe",
            )

            if self.shading == "wireframe":
                mesh.primitives[0].material.baseColorFactor = self.wireframe_opts[
                    "color"
                ]

            mesh_node = Node(mesh=mesh)
            self._scene.add_node(mesh_node)

        with warnings.catch_warnings():
            # Silence OpenGL DeprecationWarning
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            img, _ = self._renderer.render(self._scene, flags=RenderFlags.RGBA)

        self._scene.mesh_nodes.clear()
        img = torch.as_tensor(img.copy(), device=self.device)
        return img

    def close(self):
        """Closes the OffScreenRenderer object and clears the scene."""
        self._renderer.delete()
        self._scene.clear()
