import numpy as np
import mediapipe as mp


class Mediapipe:
    
    def __init__(self, static_image_mode=True, **kwargs):
        """ Initializes a Mediapipe recon model.
        
        Parameters
        ----------
        static_image_mode : bool
            Whether to expect a sequence of related images
            (like in a video)
        kwargs : dict
            Extra keyword arguments to be passed to 
            the initialization of FaceMesh
        """
        self.model = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1, refine_landmarks=True, **kwargs
        )
        self.model.__enter__()  # enter context manually

    def __call__(self, image):
        """ Performns reconstruction.
        
        Parameters
        ----------
        image : np.ndarray
            A 3D (w x h x 3) numpy array representing a RGB image
        """
        results = self.model.process(image)
            
        if not results.multi_face_landmarks:
            raise ValueError("Could not reconstruct face!")
        elif len(results.multi_face_landmarks) > 1:
            raise ValueError("Found more than 1 face!")
        else:
            lm = results.multi_face_landmarks[0].landmark
            
        x = np.array([lm_.x for lm_ in lm])
        y = np.array([lm_.y for lm_ in lm])
        z = np.array([lm_.z for lm_ in lm])
        self.v = np.c_[x, y, z]
    
    def get_v(self):
        return self.v
    
    def close(self):
        """ Closes context manager. """
        # Note: __exit__ just calls close()
        self.model.__exit__(None, None, None)