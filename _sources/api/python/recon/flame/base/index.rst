:py:mod:`medusa.recon.flame.base`
=================================

.. py:module:: medusa.recon.flame.base

.. autoapi-nested-parse::

   Module with a base class for FLAME-based reconstruction models.



Module Contents
---------------

.. py:class:: FlameReconModel



   A reconstruction model which outputs data based on the FLAME-
   topology.

   .. py:method:: is_dense()

      Checks if the current model is a dense model.

      :returns: **dense** -- True if dense, False otherwise
      :rtype: bool


   .. py:method:: get_tris()

      Retrieves the triangles (tris) associated with the predicted vertex
      mesh.


   .. py:method:: get_cam_mat()

      Returns a default camera matrix for FLAME-based reconstructions.


   .. py:method:: close()

      Sets loaded triangles to None.



