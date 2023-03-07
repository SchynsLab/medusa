:py:mod:`medusa.containers.threeD`
==================================

.. py:module:: medusa.containers.threeD

.. autoapi-nested-parse::

   Classes for representing and animating 3D face meshes (WIP).



Module Contents
---------------

.. py:data:: flame_path

   

.. py:data:: flame_generator

   

.. py:class:: Base3D

   A base class for 3D face objects.

   .. py:method:: save(path, file_type='obj', **kwargs)

      Saves a mesh to disk as an obj wavefront file.


   .. py:method:: animate(v, mat, is_deltas=True, to_4D=True)

      Animates an existing 3D mesh.



.. py:class:: Flame3D(v=None, mat=None, topo='coarse', device=DEVICE)



   A FLAME-based 3D face mesh.

   .. py:method:: from_4D(data, index=0)
      :classmethod:

      Creates a 3D object by indexing a 4D object.


   .. py:method:: random(shape=None, exp=None, pose=None, rot_x=None, rot_y=None, rot_z=None, no_exp=True)
      :classmethod:

      Creates a face with random shape/expression parametesr and pose.


   .. py:method:: animate(v, mat, sf, frame_t, is_deltas=True)

      Animates a 3D face mesh and returns a proper 4D object.



.. py:class:: Mediapipe3D(v=None, mat=None)



   A mediapipe-based 3D face mesh.

   .. py:method:: from_4D(data, index=0)
      :classmethod:

      Creates a 3D object by indexing a 4D object.



