:py:mod:`medusa.render.video`
=============================

.. py:module:: medusa.render.video

.. autoapi-nested-parse::

   A module with functionality to easily render a 4D mesh to a video using the
   ``VideoRender`` class.



Module Contents
---------------

.. py:class:: VideoRenderer(shading='flat', lights=None, loglevel='INFO')

   Renders a 4D mesh to a video.

   :param render_cls: Renderer class to use (at the moment, only PytorchRenderer is supported)
   :type render_cls: PytorchRenderer
   :param shading: Type of shading ('flat', 'smooth', or 'wireframe'; latter only when using
                   'pyrender')
   :type shading: str
   :param lights: Lights to use in rendering; if None, a default PointLight will be used
   :type lights: None, str, or pytorch3d Lights class

   .. py:method:: close()

      Closes the renderer.



