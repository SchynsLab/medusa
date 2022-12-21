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

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_cam_mat(self):
        pass

    def _check_inputs(self, inputs, expected_size, channels_first=True):

        if channels_first:
            actual_size = inputs.shape[2:]
        else:
            actual_size = inputs.shape[1:3]

        actual_size = tuple(actual_size)

        if actual_size != expected_size:
            raise ValueError(
                f"Found image(s) of size {actual_size}, but expected {expected_size}"
            )
