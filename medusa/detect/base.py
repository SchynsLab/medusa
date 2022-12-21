from pathlib import Path
from ..containers import BatchResults
from ..io import VideoLoader


class BaseDetectionModel:
    def detect_from_video(self, vid, batch_size=32):
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
            vid = VideoLoader(vid, batch_size, self.device)

        results = BatchResults(0, self.device)
        for batch in vid:
            out = self(batch)
            results.add(**out)

        results.concat()
        return results
