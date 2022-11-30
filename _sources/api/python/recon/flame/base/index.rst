:py:mod:`medusa.recon.flame.base`
=================================

.. py:module:: medusa.recon.flame.base


Module Contents
---------------

.. py:class:: FlameReconModel

   Bases: :py:obj:`medusa.recon.base.BaseReconModel`

   Base class for reconstrution models. Implements some
   abstract methods that should be implemented by classes that
   inherent from it (such as ``get_tris``) and some default
   methods (such as ``close``).

   .. py:method:: get_tris()

      Retrieves the triangles (tris) associated with the predicted vertex mesh.


   .. py:method:: close()



