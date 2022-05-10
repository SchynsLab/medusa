import numpy as np
from pathlib import Path
from trimesh import Trimesh
from pyrender.constants import RenderFlags
from pyrender import Scene, Mesh, Node, OffscreenRenderer
from pyrender import DirectionalLight, IntrinsicsCamera, OrthographicCamera


class Renderer:
    """ A high-level wrapper around a pyrender-based renderer.
    
    Parameters
    ----------
    camera_type : str
        Either 'orthographic' (for Flame-based reconstructions) or
        'intrinsic' (for mediapipe reconstruction)
    smooth : bool
        Whether to render a smooth mesh (by normal interpolation) or not
    wireframe : bool
        Whether to render a wireframe instead of a surface
    zoom_out : int/float
        How much to translate the camera into the positive z direction
        (necessary for Flame-based reconstructions)
    viewport : tuple[int]
        Desired output image size (width, height), in pixels; should match
        the original image (before cropping) that was reconstructed
    """
    
    def __init__(self, camera_type='orthographic', smooth=True, wireframe=False,
                 zoom_out=4, viewport=(224, 224)):
    
        self.camera_type = camera_type
        self.smooth = smooth
        self.wireframe = wireframe
        self.zoom_out = zoom_out
        self.viewport = viewport
        self.scene = self._create_scene()
        self._renderer = self._create_renderer()

    def _create_scene(self):
        """ Creates a simple scene with a camera and a 
        directional light. The object (reconstructed face) is
        added when calling __call__. """
        w, h = self.viewport
        scene = Scene(bg_color=[0, 0, 0, 0], ambient_light=(255, 255, 255))
        
        if self.camera_type == 'orthographic':
            camera = OrthographicCamera(xmag=1, ymag=1)
        elif self.camera_type == 'intrinsic':
            # Note to self: zfar might have to be increased if the face is
            # very small; will increase rendering time though
            camera = IntrinsicsCamera(fx=w, fy=w, cx=w/2, cy=h/2, zfar=300)
            # For mediapipe renderings, we don't need to zoom out
            self.zoom_out = 0
        
        scene.add_node(Node(camera=camera, translation=(0, 0, self.zoom_out)))
        light = DirectionalLight(intensity=5)
        scene.add_node(Node(light=light, translation=(0, 0, self.zoom_out)))

        return scene

    def _create_renderer(self):
        """ Creates the renderer with specified viewport dimensions. """
        return OffscreenRenderer(
            viewport_width=self.viewport[0], viewport_height=self.viewport[1]
        )
   
    def __call__(self, v, f):
        """ Performs the actual rendering. 
        
        Parameters
        ----------
        v : np.ndarray
            A 2D array with vertices of shape V (nr of vertices) x 3
            (X, Y, Z)
        f : np.ndarray
            A 2D array with 'faces' (triangles) of shape F (nr of faces) x 3
            (nr of vertices); should be integers
            
        Returns
        -------
        img : np.ndarray
            A 3D array (with np.uint8 integers) of shape `viewport[0]` x 
            `viewport[1]` x 3 (RGB) 
        """
        mesh = Mesh.from_trimesh(Trimesh(v, f), smooth=self.smooth, wireframe=self.wireframe)
        
        if self.wireframe:
            # Set to red, because the default isn't very well visible
            red = np.array([1., 0., 0., 1], dtype=np.float32)
            mesh.primitives[0].material.baseColorFactor = red

        mesh_node = Node(mesh=mesh)
        self.scene.add_node(mesh_node)
        img, _ = self._renderer.render(self.scene, flags=RenderFlags.RGBA)
        self.scene.remove_node(mesh_node)
        
        return img.copy()
    
    def alpha_blend(self, img, background, threshold=1):
        """ Simple alpha blend of a rendered image and
        a background. The image (`img`) is assumed to be 
        an RGBA image and the background (`background`) is
        assumed to be a RGB image. The alpha channel of the image
        is used to blend them together. The optional `threshold`
        parameter can be used to impose a sharp cutoff.
        
        Parameters
        ----------
        img : np.ndarray
            A 3D numpy array of shape height x width x 4 (RGBA)
        background : np.ndarray
            A 3D numpy array of shape height x width x 3 (RGB)
        threshold: int, float
            Threshold to impose on the alpha channel. Everything above
            this value is set to 1.
        """
        alpha = img[:, :, 3, None] / 255.
        alpha[alpha >= threshold] = 1
        img = img[:, :, :3] * alpha + (1 - alpha) * background
        
        # TODO: add global alpha level for face        
        return img.astype(np.uint8)

    def close(self):
        """ Closes the OffScreenRenderer object. """
        self._renderer.delete()
