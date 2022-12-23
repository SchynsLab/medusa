from pathlib import Path

import pytest
import torch
from conftest import _check_gha_compatible

import medusa.data
from medusa.onnx import OnnxModel


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("set_param", [False, True])
def test_onnx(device, set_param):
    if not _check_gha_compatible(device):
        return

    onnx_file = Path(medusa.data.__file__).parent / "models/yunet.onnx"

    if set_param:
        model = OnnxModel(
            onnx_file,
            device=device,
            input_names=["input"],
            input_shapes=[1, 3, 120, 160],
        )
    else:
        model = OnnxModel(onnx_file, device=device)

    inp = torch.randn(model._params["in_shapes"][0], device=device)

    outputs = model.run(inp)
    assert len(outputs) == len(model._params["out_names"])

    out0 = outputs[model._params["out_names"][0]]
    assert out0.device.type == device
    assert list(out0.shape) == model._params["out_shapes"][0]
