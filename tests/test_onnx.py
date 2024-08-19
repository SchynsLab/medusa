from pathlib import Path

import pytest
import torch
from conftest import _is_device_compatible

import medusa.data
from medusa.onnx import OnnxModel
from medusa.data import get_external_data_config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("model", ['2d106det', '1k3d68', 'genderage', 'glintr100', 'scrfd_10g_bnkps', 'yunet'])
def test_onnx(device, model):
    if not _is_device_compatible(device):
        return

    if model == 'yunet':
        onnx_file = Path(medusa.data.__file__).parent / "models/yunet.onnx"
    else:
        # isf
        onnx_file = get_external_data_config('insightface_path') / f'{model}.onnx'

    model = OnnxModel(onnx_file, device=device)
    inp_shape = model._params["in_shapes"][0]
    if inp_shape[0] == 'None':
        inp_shape[0] = 1

    if inp_shape == [1, 3, '?', '?']:
        inp_shape = [1, 3, 640, 640]

    inp = torch.randn(inp_shape, device=device)
    outputs = model.run(inp)
    assert len(outputs) == len(model._params["out_names"])

    out0 = outputs[model._params["out_names"][0]]
    assert out0.device.type == device
    assert list(out0.shape) == model._params["out_shapes"][0]
