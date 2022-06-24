import cv2
import math
import numpy as np
import torch

class PoseEstimation:
    MODEL_POINTS_SLOW = np.array([
                                (0.0, -170.0, 135.0),    # Nose tip
                                (0.0, 0.0, 0.0),         # Between the eyes
                                (0.0, -500.0, -70.0),    # Chin
                                (-225.0, 0.0, 0.0),      # Left eye left corner
                                (225.0, 0.0, 0.0),       # Right eye right corner
                                (-150.0, -320.0, 10.0),  # Left Mouth corner
                                (150.0, -320.0, 10.0)    # Right mouth corner
                                ])
    FACE_POINTS_SLOW = [30, 27, 8, 36, 45, 48, 54] # Facial landmark values for abovementioned points, as denoted by dlib

    MODEL_POINTS_FAST = np.array([
                            (0.0, -340.0, 393.0),        # Nose tip
                            (-460.0, -20.0, -26.0),      # Left eye left corner
                            (460.0, -20.0, -26.0),       # Right eye right corner
                            (-177.0, 0.0, 0.0),          # Left eye right corner
                            (177.0, 0.0, 0.0),           # Right eye left corner
                            ])
    FACE_POINTS_FAST = [4, 2, 0, 3, 1] # Facial landmark values for abovementioned points, as denoted by dlib

    def __init__(self, frame, fast = False):
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
        self.pose = None
        self.fast = fast
        if fast:
            self.model_points = self.MODEL_POINTS_FAST
            self.face_points = self.FACE_POINTS_FAST
        else:
            self.model_points = self.MODEL_POINTS_SLOW
            self.face_points = self.FACE_POINTS_SLOW

        self.height, self.width, channels = frame.shape
        center = (self.width / 2, self.height / 2)
        self.camera_matrix = np.array(
                                [[self.width, 0, center[0]],
                                [0, self.width, center[1]],
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
        self._image_points = np.array([(self.face_landmarks.part(i).x, self.face_landmarks.part(i).y) for i in self.face_points ], dtype="float32")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(self.model_points, self._image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

        # Project 3 3D point onto the image plane.
        # We use this to draw 3 lines sticking out of the nose tip
        axis = []
        if self.fast:
            axis = np.float32([[500, -340, 393],
                               [0, 160, 393],
                               [0, -340, 893]])
        else:
            axis = np.float32([[500, -170, 135],
                               [0, 330, 135],
                               [0, -170, 635]])
        (nose_point_2D, jacobian) = cv2.projectPoints(axis, rotation_vector,
                                                         translation_vector, self.camera_matrix, dist_coeffs)

        # Define the lines to be drawn
        nose = ( int(self._image_points[0][0]), int(self._image_points[0][1]))
        l1 = (int(nose_point_2D[0][0][0]), int(nose_point_2D[0][0][1]))
        l2 = (int(nose_point_2D[1][0][0]), int(nose_point_2D[1][0][1]))
        l3 = (int(nose_point_2D[2][0][0]), int(nose_point_2D[2][0][1]))
        self.lines = nose, l1, l2, l3

        # Get the projection matrix and use it to get XYZ and pitch, yaw and roll
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        self.x, self.y, self.z = self._getXYZ(proj_matrix)

        self.pitch, self.yaw, self.roll = self._rotationMatrixToEulerAngles(rotation_matrix)

        # Translate pose into a torch tensor
        self.pose = torch.tensor(np.hstack(([self.x, self.y, self.z], [self.pitch, self.yaw, self.roll])), dtype=torch.float32)

    def _rotationMatrixToEulerAngles(self, R):
        # Calculation details at https://learnopencv.com/rotation-matrix-to-euler-angles/
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def _getXYZ(self, proj_matrix):
        # Calculation details at http://faculty.salina.k-state.edu/tim/mVision/ImageFormation/projection.html
        cam_extrinsic_matrix = np.vstack([proj_matrix, [0, 0, 0, 1]])
        perspective_projection_model = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0]])
        cam_intrinsic_matrix = np.dot(self.camera_matrix, perspective_projection_model)
        camera_matrix = np.dot(cam_intrinsic_matrix, cam_extrinsic_matrix)
        point_vector = np.array([0, 0, 0, 1]) # Calculate from the point between the eyes (0,0,0 in head coordinates)
        camera_point_vector = np.dot(camera_matrix, point_vector)

        #Divide by the depth, to get the x,y coordinates without relation to depth
        depth = camera_point_vector[2]
        camera_point_vector = camera_point_vector / depth
        camera_point_vector[2] = depth

        #Scale coordinates match camera resolution
        camera_point_vector[0] /= self.width    #Scale down to be between 0 and 1
        camera_point_vector[1] /= self.height   #Scale down to be between 0 and 1
        camera_point_vector[2] /= 10000         #Scale down to be between 0 and 1

        return camera_point_vector

    def draw_facing(self, frame):
        if self.face_landmarks != None:
            nose, l1, l2, l3 = self.lines

            for p in self._image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            cv2.line(frame, nose, l1, (0, 255, 0), 3)  # GREEN
            cv2.line(frame, nose, l2, (255, 0,), 3)  # BLUE
            cv2.line(frame, nose, l3, (0, 0, 255), 3)  # RED

    def write_position_on_frame(self, frame):
        cv2.putText(frame, "Pitch:  " + str(self.pitch), (45, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Yaw: " + str(self.yaw), (45, 65), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Roll: " + str(self.roll), (45, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "X: " + str(self.x), (45, 135), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Y: " + str(self.y), (45, 170), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Z: " + str(self.z), (45, 205), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
