import cv2
import os
import dlib
import numpy as np

class FaceTracking:

    def __init__(self):
        self.frame = None
        self.faces = None
        self.landmarks = None

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    #TODO make this work
    @property
    def smooth_landmark_points(self):
        return np.mean(self.smooth_landmark_points)

    @smooth_landmark_points.setter
    def smooth_landmark_points(self, new_landmarks):
        self.smooth_landmark_points.appende(new_landmarks)
        if len(self.smooth_landmark_points) > 5:
            self.smooth_landmark_points.remove(5)


    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        return self._analyze()

    def _analyze(self):
        """Detects the face and landmarks"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self._face_detector(frame)
        try:
            self.landmarks = self._predictor(frame, self.faces[0])
        except IndexError:
            return False
        return True

    def draw_landmarks(self, frame):
        if self.landmarks != None:
            for i in range(self.landmarks.num_parts):
                            p = self.landmarks.part(i)
                            cv2.circle(frame, (p.x, p.y), 1, 255, 2)

    def draw_face_squares(self, frame):
        for f in self.faces:
            cv2.rectangle(frame, (f.left(), f.top()), (f.right(), f.bottom()), (0, 0, 255), 3)