import torch

from . import DEVICE


def sort_faces(lms_stack, img_idx, dist_threshold=200, device=DEVICE):
    face_idx = torch.zeros_like(img_idx, device=device, dtype=torch.int64)

    for i, i_img in enumerate(img_idx.unique()):
        # lms = all detected landmarks for this image
        det_idx = i_img == img_idx
        lms = lms_stack[det_idx]

        # flatten landmarks (5 x 2 -> 10)
        n_det = lms.shape[0]
        lms = lms.reshape((n_det, -1))

        if i == 0:
            # First detection, initialize tracker with first landmarks
            face_idx[det_idx] = torch.arange(0, n_det, device=device)
            tracker = lms
            continue

        # Compute the distance between each detection (lms) and currently tracked faces (tracker)
        dists = torch.cdist(lms, tracker)  # n_det x current_faces

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
        tracker[face_assigned[~unassigned]] = lms[~unassigned]

        # If there are new faces, add them to the tracker
        n_new = unassigned.sum()
        if n_new > 0:
            # Update the assigned face index for the new faces with a new integer
            face_assigned[unassigned] = tracker.shape[0] + torch.arange(n_new, device=device)
            # and add to tracker
            tracker = torch.cat((tracker, lms[unassigned]))

        # Add face selection to face_idx across images
        face_idx[det_idx] = face_assigned

    return face_idx
