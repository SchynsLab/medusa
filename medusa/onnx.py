from collections import OrderedDict

import numpy as np
import torch
from onnxruntime import InferenceSession, set_default_logger_severity

from .defaults import DEVICE


class OnnxModel:
    """Wrapper around an onnxruntime session to make running with an iobinding
    easier.

    Parameters
    ----------
    onnx_file : str, pathlib.Path
        Path to onnx file
    device : str
        Device to run model on ('cpu' or 'cuda')
    **kwargs
        Extra keyword arguments (from 'in_names', 'in_shapes', 'out_names', 'out_shapes')
        to be set in `_params` (which override those parameters from the onnx model)
    """

    def __init__(self, onnx_file, device=DEVICE, **kwargs):

        self.device = device
        self._session = self._init_session(onnx_file)
        self._binding = self._session.io_binding()
        self._params = self._extract_params(**kwargs)

    def _init_session(self, onnx_file):
        """Initializes an InferenceSession object."""
        set_default_logger_severity(3)
        device = self.device.upper()

        # per: https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459
        opts = {"cudnn_conv_algo_search": "HEURISTIC"}
        provider = [(f"{device}ExecutionProvider", opts)]
        return InferenceSession(str(onnx_file), providers=provider)

    def _extract_params(self, **kwargs):
        """Extract expected input and output names & shapes; if during
        initialization extra parameters were passed (and if they have the
        correct keys), they will override the onnx parameters."""
        params = {
            "in_names": [i_.name for i_ in self._session.get_inputs()],
            "in_shapes": [i_.shape for i_ in self._session.get_inputs()],
            "out_names": [o_.name for o_ in self._session.get_outputs()],
            "out_shapes": [o_.shape for o_ in self._session.get_outputs()],
        }

        for key, value in kwargs.items():
            if key in params:
                if not isinstance(value, list):
                    value = [value]

                exp_len = len(params[key])
                if len(value) != exp_len:
                    raise ValueError(f"Attribute {key} should have length {exp_len}!")

                params[key] = value

        return params

    def run(self, inputs, outputs_as_list=False):
        """Runs the model with given inputs.

        Parameters
        ----------
        inputs : torch.tensor, list
            Either a list of torch tensors or a single tensor (if just one input)
        outputs_as_list : bool
            Whether to return a list of outputs instead of a dict

        Returns
        -------
        outputs : OrderedDict, list
            Dictionary (or, when ``outputs_as_list``, a list) with outputs names
            and corresponding outputs (as torch tensors)
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        to_iter = zip(inputs, self._params["in_names"], self._params["in_shapes"])
        for inp, name, shape in to_iter:

            if len(inp.shape) != len(shape):
                raise ValueError(
                    f"Wrong number of dims for input {name}; expected {shape} "
                    f"but got {list(inp.shape)}!"
                )

            self._binding.bind_input(
                name=name,
                device_type=self.device,
                device_id=0,
                element_type=np.float32,
                shape=tuple(inp.shape),
                buffer_ptr=inp.data_ptr(),
            )

        outputs = OrderedDict()
        to_iter = zip(self._params["out_names"], self._params["out_shapes"])
        for name, shape in to_iter:
            outp = torch.empty(
                shape, dtype=torch.float32, device=self.device
            ).contiguous()
            self._binding.bind_output(
                name=name,
                device_type=self.device,
                device_id=0,
                element_type=np.float32,
                shape=shape,
                buffer_ptr=outp.data_ptr(),
            )
            outputs[name] = outp

        self._binding.synchronize_inputs()
        self._binding.synchronize_outputs()
        self._session.run_with_iobinding(self._binding)

        if outputs_as_list:
            outputs = list(outputs.values())

        return outputs
