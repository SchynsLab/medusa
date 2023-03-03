:py:mod:`medusa.recon.flame.mica.recon`
=======================================

.. py:module:: medusa.recon.flame.mica.recon

.. autoapi-nested-parse::

   An implementation of the MICA reconstruction model.

   For the associated license, see license.md.



Module Contents
---------------

.. py:class:: MicaReconModel(device=DEVICE)



   A simplified implementation of the MICA 3D face reconstruction model
   (https://zielon.github.io/mica/), for inference only.

   :param device: Either 'cuda' (uses GPU) or 'cpu'
   :type device: str

   .. py:method:: get_cam_mat()

      Gets a default camera matrix that most likely renders a face in full
      view.
