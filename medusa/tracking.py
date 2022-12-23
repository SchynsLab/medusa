"""Module with functionality to 'track' faces, i.e., to associate the same face
across detections/reconstructions from multiple consecutive (video) frames."""
import torch


def sort_faces(lms, img_idx, dist_threshold=250):
    """'Sorts' faces across multiple frames.

    Parameters
    ----------
    lms : torch.tensor
        A float tensor of shape *B* (batch size) x *V* (vertices/landmarks) x *C*
        (coordinates), which will be used to sort the faces
    img_idx : torch.tensor
        An integer tensor with the image index associated with each detection
        (e.g., [0, 0, 1, 1, 1, ...] means that there are two faces in the first image,
        three faces in the second image, etc.)
    dist_threshold : torch.tensor
        Euclidean distance between two sets of landmarks/vertices that we consider
        comes from two different faces (e.g., if ``d(lms1, lms2) >= dist_treshold``,
        then we conclude that face 1 (``lms1``) is a different from face 2 (``lms2``)
    present_treshold : float
        Any face that is tracked less than a ``present_treshold`` proportion of frames
        will not be included in the selection (i.e., in ``keep``)

    Returns
    -------
    face_idx : torch.tensor
        An integer tensor of length *n detections*, in which each unique value
        represents a unique face
    keep : torch.tensor
        A boolean tensor of length *n detections*, in which each value indicates
        whether that detection should be kept (``True``) or discarded (``False``)
    """
    device = lms.device
    face_idx = torch.zeros_like(img_idx, device=device, dtype=torch.int64)

    for i, i_img in enumerate(img_idx.unique()):
        # lms = all detected landmarks/vertices for this image
        det_idx = i_img == img_idx
        lms_img = lms[det_idx]

        # flatten landmarks (5 x 2 -> 10)
        n_det = lms_img.shape[0]
        lms_img = lms_img.reshape((n_det, -1))

        if i == 0:
            # First detection, initialize tracker with first landmarks
            face_idx[det_idx] = torch.arange(0, n_det, device=device)
            tracker = lms_img.clone()
            continue

        # Compute the distance between each detection (lms) and currently tracked faces (tracker)
        dists = torch.cdist(lms_img, tracker)  # n_det x current_faces

        # face_assigned keeps track of which detection is assigned to which face from
        # the tracker (-1 refers to "not assigned yet")
        face_assigned = torch.ones(n_det, device=device, dtype=torch.int64) * -1

        # We'll keep track of which tracked faces we have not yet assigned
        track_list = torch.arange(dists.shape[1], device=device)

        # Check the order of minimum distances across detections
        # (which we'll use to loop over)
        order = dists.min(dim=1)[0].argsort()
        det_list = torch.arange(dists.shape[0], device=device)

        # Loop over detections, sorted from best match (lowest dist)
        # to any face in the tracker to worst match
        for i_det in det_list[order]:

            if dists.shape[1] == 0:
                # All faces from tracker have been assigned!
                # So this detection must be a new face
                continue

            # Extract face index with the minimal distance (`min_face`) ...
            min_dist, min_face = dists[i_det, :].min(dim=0)

            # And check whether it is acceptably small
            if min_dist < dist_threshold:

                # Assign to face_assigned; note that we cannot use `min_face`
                # directly, because we'll slice `dists` below
                face_assigned[i_det] = track_list[min_face]

                # Now, for some magic: remove the selected face
                # from dists (and track_list), which will make sure
                # that the next detection cannot be assigned the same
                # face
                keep = track_list != track_list[min_face]
                dists = dists[:, keep]
                track_list = track_list[keep]
            else:
                # Detection cannot be assigned to any face in the tracker!
                # Going to update tracker with this detection
                pass

        # Update the tracker with the (assigned) detected faces
        unassigned = face_assigned == -1
        tracker[face_assigned[~unassigned]] = lms_img[~unassigned]

        # If there are new faces, add them to the tracker
        n_new = unassigned.sum()
        if n_new > 0:
            # Update the assigned face index for the new faces with a new integer
            face_assigned[unassigned] = tracker.shape[0] + torch.arange(
                n_new, device=device
            )
            # and add to tracker
            tracker = torch.cat((tracker, lms_img[unassigned]))

        # Add face selection to face_idx across images
        face_idx[det_idx] = face_assigned

    return face_idx


def filter_faces(face_idx, n_img, present_threshold=0.1):
    """Function to filter faces based on various criteria.

    For now, only filters based on how frequent a face is detected
    across frames.
    """
    # Loop over unique faces tracked
    keep = torch.full_like(face_idx, fill_value=True, dtype=torch.bool)
    for f in face_idx.unique():
        # Compute the proportion of images containing this face
        f_idx = face_idx == f
        prop = (f_idx).sum() / n_img
        if prop < present_threshold:
            keep[f_idx] = False

    return keep


def _ensure_consecutive_face_idx(face_idx):
    """Makes sure that face IDs in ``face_idx`` are always consecutive, and if
    not, makes it so.

    For example, if the unique IDs are ``[0, 1, 5]``, it will change it
    to ``[0, 1, 2]``.
    
    Parameters
    ----------
    face_idx : torch.tensor
        Tensor with integers corresponding to each detection's face ID

    Returns
    -------
    new_face_idx : torch.tensor
        Tensor with integers corresponding to each detection's face ID,
        but now those integers are guaranteed to be consecutive
    """

    ids = face_idx.unique()
    if (ids.diff() == 1).all():
        # All consecutive, so just return face_idx
        return face_idx
    else:
        # Fix
        new_face_idx = face_idx.clone()
        for i, id_ in enumerate(ids):
            new_face_idx[face_idx == id_] = i

        return new_face_idx
