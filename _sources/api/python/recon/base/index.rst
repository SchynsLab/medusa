:py:mod:`medusa.recon.base`
===========================

.. py:module:: medusa.recon.base


Module Contents
---------------

.. py:class:: BaseReconModel

   Bases: :py:obj:`abc.ABC`

   Base class for reconstrution models. Implements some
   abstract methods that should be implemented by classes that
   inherent from it (such as ``get_tris``) and some default
   methods (such as ``close``).

   .. py:method:: get_tris()
      :abstractmethod:


   .. py:method:: close()



