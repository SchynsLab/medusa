import pytest
import torch

from medusa.detect import SCRFDetector
from medusa.tracking import (_ensure_consecutive_face_idx, filter_faces,
                             sort_faces)


@pytest.mark.parametrize("video_test", [1, 2, 3, 4], indirect=True)
def test_tracking_from_detections(video_test):
    detector = SCRFDetector()
    results = detector.detect_faces_video(video_test)

    face_idx = sort_faces(results.lms, results.img_idx, dist_threshold=250)

    # Remove infrequent (<10%) faces
    keep = filter_faces(face_idx, results.n_img, present_threshold=0.1)
    face_idx = face_idx[keep]

    # Check if number of expected faces matches the actual number of faces
    n_exp = int(video_test.stem[0])
    assert face_idx.unique().numel() == n_exp

    # Make sure face IDs are consecutive (not strictly necessary, but nice to have)
    face_idx = _ensure_consecutive_face_idx(face_idx)

    # Check if actually consecutive
    n_faces = face_idx.unique().numel()
    torch.testing.assert_close(
        face_idx.unique(), torch.arange(n_faces, device=face_idx.device)
    )
