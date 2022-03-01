import cv2
import math
import numpy as np

class PoseEstimation:

    MODEL_POINTS = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
    FACE_POINTS = [30, 8, 36, 45, 48, 54]

    def __init__(self, frame):
        self.x = None
        self.y = None
        self.z = None
        self.pitch = None
        self.yaw = None
        self.roll = None
        self.face_landmarks = None
        self._image_points = None
        self.lines = None
        self.frame = frame

        size = frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        self.camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )

    def _get_2d_points(self, img, rotation_vector, translation_vector, camera_matrix, val):
        """Return the 3D points present as 2D for making annotation box"""
        point_3d = []
        dist_coeffs = np.zeros((4,1))
        rear_size = val[0]
        rear_depth = val[1]
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))
        
        front_size = val[2]
        front_depth = val[3]
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
        
        # Map to 2d img points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                        rotation_vector,
                                        translation_vector,
                                        camera_matrix,
                                        dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        return point_2d

    def _draw_annotation_box(self, img, rotation_vector, translation_vector, camera_matrix,
                            rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                            color=(255, 255, 0), line_width=2):
        """
        Draw a 3D anotation box on the face for head pose estimation
        Parameters
        ----------
        img : np.unit8
            Original Image.
        rotation_vector : Array of float64
            Rotation Vector obtained from cv2.solvePnP
        translation_vector : Array of float64
            Translation Vector obtained from cv2.solvePnP
        camera_matrix : Array of float64
            The camera matrix
        rear_size : int, optional
            Size of rear box. The default is 300.
        rear_depth : int, optional
            The default is 0.
        front_size : int, optional
            Size of front box. The default is 500.
        front_depth : int, optional
            Front depth. The default is 400.
        color : tuple, optional
            The color with which to draw annotation box. The default is (255, 255, 0).
        line_width : int, optional
            line width of lines drawn. The default is 2.
        Returns
        -------
        None.
        """
        
        rear_size = 1
        rear_depth = 0
        front_size = img.shape[1]
        front_depth = front_size*2
        val = [rear_size, rear_depth, front_size, front_depth]
        point_2d = self._get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
        # # Draw all the lines
        cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
        
        
    def _head_pose_points(self, img, rotation_vector, translation_vector, camera_matrix):
        """
        Get the points to estimate head pose sideways    
        Parameters
        ----------
        img : np.unit8
            Original Image.
        rotation_vector : Array of float64
            Rotation Vector obtained from cv2.solvePnP
        translation_vector : Array of float64
            Translation Vector obtained from cv2.solvePnP
        camera_matrix : Array of float64
            The camera matrix
        Returns
        -------
        (x, y) : tuple
            Coordinates of line to estimate head pose
        """
        rear_size = 1
        rear_depth = 0
        front_size = img.shape[1]
        front_depth = front_size*2
        val = [rear_size, rear_depth, front_size, front_depth]
        point_2d = self._get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
        y = (point_2d[5] + point_2d[8])//2
        x = point_2d[2]
        
        return (x, y)

    def refresh(self, frame, landmarks):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self.face_landmarks = landmarks
        self._analyze()

    def _analyze(self):
        self._image_points = np.array([(self.face_landmarks.part(i).x, self.face_landmarks.part(i).y) for i in self.FACE_POINTS ], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.MODEL_POINTS, self._image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, self.camera_matrix, dist_coeffs)
        
        p1 = ( int(self._image_points[0][0]), int(self._image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = self._head_pose_points(self.frame, rotation_vector, translation_vector, self.camera_matrix)

        self.lines = p1, p2, x1, x2

        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            self.pitch = int(math.degrees(math.atan(m)))
        except:
            self.pitch = 90
            
        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            self.yaw = int(math.degrees(math.atan(-1/m)))
        except:
            self.yaw = 90

    def draw_facing(self, frame):
        p1, p2, x1, x2 = self.lines

        for p in self._image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        cv2.line(frame, tuple(x1), tuple(x2), (255, 255, 0), 2)

    def write_position_on_frame(self, frame):
        cv2.putText(frame, "Pitch:  " + str(self.pitch), (45, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Yaw: " + str(self.yaw), (45, 65), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
