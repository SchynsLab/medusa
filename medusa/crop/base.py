from pathlib import Path
from ..containers import BatchResults
from ..io import VideoLoader


class BaseCropModel:
    """Base crop model, from which all crop models inherit."""

    def crop_faces_video(self, vid, batch_size=32, save_imgs=False):
        """Utility function to crop all faces in each frame of a video.

        Parameters
        ----------
        vid : str, Path
            Path to video (or, optionally, a ``VideoLoader`` object)

        Returns
        -------
        results : BatchResults
            A BatchResults object with all crop information/results
        """
        if isinstance(vid, (str, Path)):
            vid = VideoLoader(vid, batch_size, self.device)

        results = BatchResults(0, self.device)
        for batch in vid:
            out = self(batch)
            results.add(**out)

            if save_imgs:
                results.add(imgs=batch)

        results.concat()
        return results