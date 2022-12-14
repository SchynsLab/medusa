from abc import ABC, abstractmethod

from ..io import load_inputs


class BaseReconModel(ABC):
    """Base class for reconstruction models.

    Implements some abstract methods that should be implemented by
    classes that inherent from it (such as ``get_tris``) and some
    default methods (such as ``close``).
    """
    @abstractmethod
    def get_tris(self):
        pass

    def close(self):
        pass

    def _load_inputs(self, inputs, *args, **kwargs):
        """Loads and checks inputs."""
        return load_inputs(inputs, *args, **kwargs)

    def _check_inputs(self, inputs, expected_size, channels_first=True):

        if channels_first:
            actual_size = inputs.shape[2:]
        else:
            actual_size = inputs.shape[1:3]

        actual_size = tuple(actual_size)

        if actual_size != expected_size:
            raise ValueError(f"Found image(s) of size {actual_size}, but expected {expected_size}")
