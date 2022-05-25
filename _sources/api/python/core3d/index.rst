:py:mod:`medusa.core3d`
=======================

.. py:module:: medusa.core3d


Module Contents
---------------

.. py:class:: Base3D

   .. py:method:: save_obj(self)


   .. py:method:: render_image(self, f_out=None)


   .. py:method:: animate(self)



.. py:class:: Flame3D(v=None, mat=None)

   Bases: :py:obj:`Base3D`

   .. py:method:: from_4D(cls, data, index=0)
      :classmethod:



.. py:class:: Mediapipe3D(v=None, mat=None)

   Bases: :py:obj:`Base3D`

   .. py:method:: from_4D(cls, data, index=0)
      :classmethod:


   .. py:method:: animate(self, v, mat, sf, frame_t, is_deltas=True)



