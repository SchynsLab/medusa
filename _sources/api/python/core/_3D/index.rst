:orphan:

:py:mod:`medusa.core._3D`
=========================

.. py:module:: medusa.core._3D


Module Contents
---------------

.. py:class:: Base3D

   .. py:method:: save(path, file_type='obj', **kwargs)


   .. py:method:: animate(v, mat, is_deltas=True)



.. py:class:: Flame3D(v=None, mat=None, dense=False)

   Bases: :py:obj:`Base3D`

   .. py:method:: from_4D(data, index=0)
      :classmethod:


   .. py:method:: animate(v, mat, sf, frame_t, is_deltas=True)



.. py:class:: Mediapipe3D(v=None, mat=None)

   Bases: :py:obj:`Base3D`

   .. py:method:: from_4D(data, index=0)
      :classmethod:


   .. py:method:: animate(v, mat, sf, frame_t, is_deltas=True)



