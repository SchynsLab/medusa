"""Module with functionality to render 4D face mesh data.

The ``Renderer`` class is a high-level wrapper around functionality from the
excellent `pyrender <https://pyrender.readthedocs.io>`_ package [1]_.

.. [1] Matl, Matthew. *pyrender* [computer software]. https://github.com/mmatl/pyrender
"""

import numpy as np
from pyrender import (DirectionalLight, IntrinsicsCamera, Mesh, Node,
                      OffscreenRenderer, OrthographicCamera, Scene)
from pyrender.constants import RenderFlags
from trimesh import Trimesh


class Renderer:
    """A high-level wrapper around a pyrender-based renderer.

    Parameters
    ----------
    viewport : tuple[int]
        Desired output image size (width, height), in pixels; should match
        the original image (before cropping) that was reconstructed
    camera_type : str
        Either 'orthographic' (for Flame-based reconstructions) or
        'intrinsic' (for mediapipe reconstruction, i.e., a perspective camera)
    smooth : bool
        Whether to render a smooth mesh (by normal interpolation) or not
    wireframe : bool
        Whether to render a wireframe instead of a surface
    """

    def __init__(self, viewport, camera_type="orthographic", smooth=True,
                 wireframe=False, cam_mat=None, focal_length=None):

        self.viewport = viewport
        self.camera_type = camera_type
        self.smooth = smooth
        self.wireframe = wireframe
        self.cam_mat = cam_mat
        self.focal_length = focal_length
        self.scene = self._create_scene()
        self._renderer = self._create_renderer()  # actual pyrender renderer

    def _create_scene(self):
        """Creates a simple scene with a camera and a directional light.

        The object (reconstructed face) is added when calling __call__.
        """
        w, h = self.viewport
        scene = Scene(bg_color=[0, 0, 0, 0], ambient_light=(255, 255, 255))

        if self.camera_type == "orthographic":
            camera = OrthographicCamera(xmag=1, ymag=1)
        elif self.camera_type == "intrinsic":
            # Note to self: zfar might have to be increased if the face is
            # very small; will increase rendering time though
            if self.focal_length is None:
                fx, fy = w, w
            else:
                fx, fy = self.focal_length, self.focal_length

            camera = IntrinsicsCamera(fx=fx, fy=fy, cx=w / 2, cy=h / 2, zfar=300)

        camera_node = Node(camera=camera, matrix=self.cam_mat)
        scene.add_node(camera_node)
        light = DirectionalLight(intensity=5)
        scene.add_node(Node(light=light, matrix=self.cam_mat))

        return scene

    def _create_renderer(self):
        """Creates the renderer with specified viewport dimensions."""
        return OffscreenRenderer(viewport_width=self.viewport[0],
                                 viewport_height=self.viewport[1])

    def __call__(self, v, f, overlay=None, cmap_name='bwr', is_colors=False):
        """Performs the actual rendering.

        Parameters
        ----------
        v : np.ndarray
            A 2D array with vertices of shape V (nr of vertices) x 3
            (X, Y, Z)
        f : np.ndarray
            A 2D array with 'faces' (triangles) of shape F (nr of faces) x 3
            (nr of vertices); should be integers
        overlay : np.ndarray
            A 1D array with overlay values (numpy floats between 0 and 1), one for each
            vertex or face
        cmap_name : str
            Name of (matplotlib) colormap; only relevant if ``overlay`` is not ``None``
        is_colors : bool
            If ``True``, then ``overlay`` is a V (of F) x 4 (RGBA) array; if ``False``,
            ``overlay`` is assumed to be a 1D array with floats betwee 0 and 1

        Returns
        -------
        img : np.ndarray
            A 3D array (with np.uint8 integers) of shape ``viewport[0]`` x
            ``viewport[1]`` x 3 (RGB)
        """

        # if overlay is not None:
        #     overlay = overlay.squeeze()

        #     if overlay.ndim > 1:
        #         raise ValueError("Overlay should be a 1D array!")

        #     if overlay.size != v.shape[0]:
        #         raise ValueError("Overlay should have the same number as values as "
        #                          f"the vertex array (expected {v.shape[0]}, found {overlay.size})!")

        #     cmap = plt.get_cmap(cmap_name)
        #     overlay = cmap(overlay)
        #     #mesh.visual.vertex_colors = cmap(overlay)

        kwargs = {}
        if overlay is not None:
            if overlay.shape[0] == f.shape[0]:
                kwargs['face_colors'] = overlay
            elif overlay.shape[0] == v.shape[0]:
                kwargs['vertex_colors'] = overlay
            else:
                raise ValueError("Cannot infer whether overlay refers to vertices or "
                                 "faces (polygons)!")

            #if not is_colors:
            #    cmap = plt.get_cmap(cmap_name)

        mesh = Trimesh(v, f, **kwargs)
        mesh = Mesh.from_trimesh(mesh, smooth=self.smooth, wireframe=self.wireframe)

        if self.wireframe:
            # Set to red, because the default isn't very well visible
            red = np.array([1.0, 0.0, 0.0, 1], dtype=np.float32)
            mesh.primitives[0].material.baseColorFactor = red

        mesh_node = Node(mesh=mesh)
        self.scene.add_node(mesh_node)
        img, _ = self._renderer.render(self.scene, flags=RenderFlags.RGBA)
        self.scene.remove_node(mesh_node)

        return img.copy()

    def alpha_blend(self, img, background, face_alpha=None):
        """Simple alpha blend of a rendered image and a background. The image
        (`img`) is assumed to be an RGBA image and the background
        (`background`) is assumed to be a RGB image. The alpha channel of the
        image is used to blend them together. The optional `threshold`
        parameter can be used to impose a sharp cutoff.

        Parameters
        ----------
        img : np.ndarray
            A 3D numpy array of shape height x width x 4 (RGBA)
        background : np.ndarray
            A 3D numpy array of shape height x width x 3 (RGB)
        """
        alpha = img[:, :, 3, None] / 255.0

        if face_alpha is not None:
            alpha[alpha > face_alpha] = face_alpha

        img = img[:, :, :3] * alpha + (1 - alpha) * background

        # TODO: add global alpha level for face
        return img.astype(np.uint8)

    def close(self):
        """Closes the OffScreenRenderer object."""
        self._renderer.delete()
