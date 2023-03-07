:py:mod:`medusa.onnx`
=====================

.. py:module:: medusa.onnx

.. autoapi-nested-parse::

   Module with a class to make working with ONNX models easier.



Module Contents
---------------

.. py:class:: OnnxModel(onnx_file, device=DEVICE, **kwargs)

   Wrapper around an onnxruntime session to make running with an iobinding
   easier.

   :param onnx_file: Path to onnx file
   :type onnx_file: str, pathlib.Path
   :param device: Device to run model on ('cpu' or 'cuda')
   :type device: str
   :param \*\*kwargs: Extra keyword arguments (from 'in_names', 'in_shapes', 'out_names', 'out_shapes')
                      to be set in `_params` (which override those parameters from the onnx model)

   .. py:method:: run(inputs, outputs_as_list=False)

      Runs the model with given inputs.

      :param inputs: Either a list of torch tensors or a single tensor (if just one input)
      :type inputs: torch.tensor, list
      :param outputs_as_list: Whether to return a list of outputs instead of a dict
      :type outputs_as_list: bool

      :returns: **outputs** -- Dictionary (or, when ``outputs_as_list``, a list) with outputs names
                and corresponding outputs (as torch tensors)
      :rtype: OrderedDict, list



