import cv2
import os
import dlib
import numpy as np

class FaceTracking:

    def __init__(self):
        self.frame = None
        self.faces = None
        self.landmarks = None
        self.lastXLandmarks = dlib.full_object_detections()
        self.lastXLandmarksAverage = None

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)


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
            if self.lastXLandmarks.__len__() >= 5:
                self.lastXLandmarks.pop(0)
            self.lastXLandmarks.append(self.landmarks)
            self.calculateAverageLandmark()
        except IndexError:
            return False
        return True

    def calculateAverageLandmark(self):
        if self.landmarks != None:
            i = 0
            rl, rr, rt, rb = 0, 0, 0, 0
            parts = dlib.points()
            for landmarks in self.lastXLandmarks:
                i = i + 1
                rl, rr, rt, rb = rl + landmarks.rect.left(), \
                                 rr + landmarks.rect.right(), \
                                 rt + landmarks.rect.top(), \
                                 rb + landmarks.rect.bottom()
                if i == 1:
                    for j in range(landmarks.num_parts):
                        parts.append(landmarks.part(j))
                else:
                    for j in range(landmarks.num_parts):
                        part = parts.pop(0)
                        new_part = dlib.point(part.x + landmarks.part(j).x, part.y + landmarks.part(j).y)
                        parts.append(new_part)

            rl, rr, rt, rb = int(rl / i), int(rr / i), int(rt / i), int(rb / i)
            rect = dlib.rectangle(rl, rt, rr, rb)

            for j in range(self.landmarks.num_parts):
                part = parts.pop(0)
                new_part = dlib.point(int(part.x / i), int(part.y / i))
                parts.append(new_part)

            self.lastXLandmarksAverage = dlib.full_object_detection(rect, parts)

    def draw_landmarks(self, frame):
        if self.landmarks != None:
            for i in range(self.landmarks.num_parts):
                p = self.landmarks.part(i)
                cv2.circle(frame, (p.x, p.y), 1, 255, 2)

        if self.lastXLandmarksAverage != None:
            for i in range(self.lastXLandmarksAverage.num_parts):
                p = self.lastXLandmarksAverage.part(i)
                cv2.circle(frame, (p.x, p.y), 1, 0, 2)

    def draw_face_squares(self, frame):
        for f in self.faces:
            cv2.rectangle(frame, (f.left(), f.top()), (f.right(), f.bottom()), (0, 0, 255), 3)