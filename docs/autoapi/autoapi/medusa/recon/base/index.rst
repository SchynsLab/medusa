:py:mod:`medusa.recon.base`
===========================

.. py:module:: medusa.recon.base

.. autoapi-nested-parse::

   Module with a base reconstruction model class.



Module Contents
---------------

.. py:class:: BaseReconModel



   Base class for reconstruction models.

   Implements some abstract methods that should be implemented by
   classes that inherent from it (such as ``get_tris``) and some
   default methods (such as ``close``).

   .. py:method:: get_tris()
      :abstractmethod:


   .. py:method:: close()
      :abstractmethod:


   .. py:method:: get_cam_mat()
      :abstractmethod:
