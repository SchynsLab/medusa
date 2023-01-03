"""A module with functionality to perform Action Units (AU) decoding from 4D
reconstructions (WIP)."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from medusa.preproc import align


class ActionUnitDecoder:
    """Class to perform Action Unit decoding from meshes.
    
    Parameters
    ----------
    recon_model : str
        Name of the reconstruction model
    predict_amplitude : bool
        Whether to predict amplitude (``True``) or only perform detection (``False``)
    """
    def __init__(self, recon_model=None, predict_amplitude=True):
        """Initializes an ActionUnitDecoder object."""
        self.recon_model = recon_model
        self.predict_amplitude = predict_amplitude
        self._detect_model = None
        self._amp_model = None
        self._setup()

    def _setup(self):
        """Sets up detection and/or amplitude model."""
        if self.recon_model is not None:
            self._setup_detect_model(self.recon_model)

            if self.predict_amplitude:
                self._setup_amp_model(self.recon_model)

    def _setup_detect_model(self, recon_model):
        """Sets up detection model."""
        f_in = (
            Path(__file__).parent
            / f"data/au/model-{recon_model}_mode-detect_model.onnx"
        )
        self._detect_model = ort.InferenceSession(str(f_in))
        f_in = (
            Path(__file__).parent
            / f"data/au/model-{recon_model}_mode-detect_thresholds.npy"
        )
        self._detect_thresh = np.load(f_in)

    def _setup_amp_model(self, recon_model):
        """Sets up amplitude model."""
        pass

    def decode(self, data, neutral_frame=0):
        """Decodes AUs from a ``Data4D`` object."""
        if data.space == "world":
            data = align(data)

        inp_name = self._detect_model.get_inputs()[0].name

        neutral = data.v[neutral_frame, ...]
        neu_mu = neutral.mean(axis=0)[None, None, :]
        neu_sd = neutral.std(axis=0)[None, None, :]

        v = (data.v.copy() - neu_mu) / neu_sd
        v = v.reshape((v.shape[0], -1))

        for i in range(len(data)):
            p_au = self._detect_model.run(None, {inp_name: v[None, i, ...]})[
                0
            ].squeeze()
