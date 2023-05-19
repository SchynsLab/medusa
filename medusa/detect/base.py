from abc import abstractmethod
from pathlib import Path
from torch import nn

from ..containers import BatchResults
from ..io import VideoLoader


class BaseDetector(nn.Module):
    """Base class for face detection models."""

    def detect_faces_video(self, vid, batch_size=32):
        """Utility function to get all detections in a video.

        Parameters
        ----------
        vid : str, Path
            Path to video (or, optionally, a ``VideoLoader`` object)

        Returns
        -------
        results : BatchResults
            A BatchResults object with all detection information
        """
        if isinstance(vid, (str, Path)):
            vid = VideoLoader(vid, batch_size=batch_size, device=self.device)

        results = BatchResults(0, self.device)
        for batch in vid:
            out = self(batch.to(self.device))
            results.add(**out)

        results.concat()
        return results
