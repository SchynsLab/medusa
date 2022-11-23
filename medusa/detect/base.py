import cv2
from pathlib import Path

from ..io import load_inputs


class BaseDetectionModel:

    @staticmethod
    def visualize(image, bbox, lms=None, conf=None, f_out=None):
        """ Creates an image with the estimated bounding box (bbox) on top of it.
        
        Parameters
        ----------
        image : array_like
            A numpy array with the original (uncropped images); can also be
            a torch Tensor; can be a batch of images or a single image
        bbox : np.ndarray
            A numpy array with the bounding box(es) corresponding to the
            image(s)
        f_out : str, pathlib.Path
            If multiple images, a number (_xxx) is appended
        """

        if f_out is not None:
            if not isinstance(f_out, Path):
                f_out = Path(f_out)

        # batch_size x h x w x 3
        image = load_inputs(image, load_as='numpy', channels_first=False, device='cpu')

        n_images = image.shape[0]
        if n_images != len(bbox):
            raise ValueError(f"Number of images ({n_images}) is not the same as "
                             f"the number of bboxes ({len(bbox)})!")
        
        if f_out is None:
            to_return = []

        for i_img in range(n_images):
            
            image_ = cv2.cvtColor(image[i_img, ...], cv2.COLOR_RGB2BGR)

            if bbox[i_img] is None:
                n_face = 0
            else:
                n_face = bbox[i_img].shape[0]
            
            for i_face in range(n_face):
                bbox_ = bbox[i_img][i_face, ...].round().astype(int)
                # color is actually red, because image_ is BGR
                cv2.rectangle(image_, bbox_[:2], bbox_[2:], (0, 0, 255), 2)

                if conf is not None:
                    conf_ = str(round(conf[i_img][i_face], 2))
                    pos = (bbox_[0], int(bbox_[1] - 5))
                    cv2.putText(image_, conf_, pos, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

                if lms is not None:
                    lms_ = lms[i_img].round().astype(int)
                    for i_lm in range(lms_.shape[1]):  # probably 5 lms
                        cv2.circle(image_, lms_[i_face, i_lm, :], 3, (0, 255, 0), -1)

            if f_out is None:
                to_return.append(image_)
            else:
                if image.shape[0] == 1:
                    this_f_out = f_out
                else:
                    this_f_out = f_out.parent / f_out.stem + f'_{i_img:02d}{f_out.suffix}'

                if this_f_out.is_file():
                    this_f_out.unlink()

                # Note to self: image_ is already BGR (see top of loop)
                success = cv2.imwrite(str(this_f_out), image_)
                if not success:
                    raise ValueError(f"Something went wrong trying to save {str(this_f_out)}!")

        if f_out is None:
            return to_return