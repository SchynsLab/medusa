:py:mod:`medusa.core3d`
=======================

.. py:module:: medusa.core3d


Module Contents
---------------

.. py:class:: Base3D

   .. py:method:: save_obj()


   .. py:method:: render_image(f_out=None)


   .. py:method:: animate()



.. py:class:: Flame3D(v=None, mat=None)

   Bases: :py:obj:`Base3D`

   .. py:method:: from_4D(data, index=0)
      :classmethod:



.. py:class:: Mediapipe3D(v=None, mat=None)

   Bases: :py:obj:`Base3D`

   .. py:method:: from_4D(data, index=0)
      :classmethod:


   .. py:method:: animate(v, mat, sf, frame_t, is_deltas=True)



