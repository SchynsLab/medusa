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


   .. py:method:: forward(imgs)

      Performs 3D reconstruction on the supplied image.

      :param imgs: Ideally, a numpy array or torch tensor of shape 1 x 3 x 112 x 112
                   (1, C, W, H), representing a cropped image as done by the
                   InsightFaceCroppingModel
      :type imgs: np.ndarray, torch.Tensor

      :returns: **out** -- A dictionary with two keys: ``"v"``, the reconstructed vertices (5023 in
                total) and ``"mat"``, a 4x4 Numpy array representing the local-to-world
                matrix, which is in the case of MICA the identity matrix
      :rtype: dict



