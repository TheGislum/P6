import cv2
import torch
import mouse
import ctypes
import keyboard
from pose_estimation import PoseEstimation
from face_tracking import FaceTracking, FastFaceTracking
from eye_isolation import EyeIsolation, FastEyeIsolation

class DataCollector:
    def __init__(self, calibration = False, fast=False, model_colour = False, camera_resolution=10000, version = 2, verbose=False):
        if calibration:
            self.dataset_dir = "./calibration_dataset/"
            self.stepsX = 2
            self.stepsY = 2
            self.x_duration = 1.5
        else:
            self.dataset_dir = "./eye_dataset/"
            self.stepsX = 8
            self.stepsY = 8
            self.x_duration = (80/120)*0.5
        self.collected = 0
        self.left_eye_list = []
        self.right_eye_list = []
        self.lable_list = []
        self.pose_list = []
        self.marginX = 40
        self.marginY = 20
        self.fast = fast
        self.model_colour = model_colour
        self.version = version
        self.verbose = verbose
        self.webcam = cv2.VideoCapture(0)

        if camera_resolution > 1080:
            camera_resolution = (10000, 10000)
        elif camera_resolution > 720:
            camera_resolution = (1920, 1080)
        elif camera_resolution > 480:
            camera_resolution = (1280, 720)
        else:
            camera_resolution = (720, 480)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])     #Set resolution of device
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
        ret, frame = self.webcam.read()

        if fast:
            self.face = FastFaceTracking()
        else:
            self.face = FaceTracking()
        
        self.pose = PoseEstimation(frame=frame, fast=fast)

    def RunCollection(self):
        user32 = ctypes.windll.user32
        screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) # get monitor resolution
        #screenWidth, screenHeight = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79) # get multi monitor resolution (all monitors)
        stop = False
        j = 0

        mouse.move(self.marginX, self.marginY)

        if self.verbose:
            print("Look at cursor while calibrating")
            print("To start calibration press 's', to cancel press 'c'")

        while keyboard.is_pressed('s') != True:
            i = 0

        stepX = int((screenWidth - (self.marginX*2)) / (self.stepsX-1))
        stepY = int((screenHeight - (self.marginY*2)) / (self.stepsY-1))

        for i in range(self.marginX, (screenWidth - self.marginX)+1, stepX):
            if stop:
                break
            if i != self.marginX:
                mouse.move(i, self.marginY, duration=1.5)
            for j in range(self.marginY, (screenHeight - self.marginY)+1, stepY):
                mouse.move(i, j, duration=self.x_duration) # steps_per_second=120
                screenX, screenY = mouse.get_position()
                position = self.GetPosition(screenX, screenWidth, screenY, screenHeight)

                if keyboard.is_pressed('c') == True:
                    stop = True
                if stop:
                    break

                self.collected = 0

                while self.collected <= 3:
                    # We get a new frame from the webcam
                    self.GetNewFrame(position)

        if self.verbose:
            print("Calibration done, saving data...")

        self.SaveData()

    def GetPosition(self, screenX, screenWidth, screenY, screenHeight):
        x = (screenX/(screenWidth/2))-1
        y = 1-(screenY/(screenHeight/2))
        position = torch.tensor([x, y], dtype=torch.float)
        return position

    def GetNewFrame(self, position):
        ret, frame = self.webcam.read()
        if ret == True:
            if self.face.refresh(frame):
                self.collected += 1
                if self.fast:
                    left_eye = FastEyeIsolation(frame, self.face.landmarks, 0, (60, 36)).colour_frame
                    right_eye = FastEyeIsolation(frame, self.face.landmarks, 1, (60, 36)).colour_frame
                else:
                    left_eye = EyeIsolation(frame, self.face.landmarks, 0, (60, 36)).colour_frame
                    right_eye = EyeIsolation(frame, self.face.landmarks, 1, (60, 36)).colour_frame

                if self.version == 3:
                    self.pose.refresh(frame, self.face.landmarks)
                    self.pose_list.append(self.pose.pose)

                if self.model_colour:
                    self.left_eye_list.append(torch.tensor(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0))
                    self.right_eye_list.append(torch.tensor(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0))
                else:
                    self.left_eye_list.append(torch.tensor(cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)).unsqueeze(0))
                    self.right_eye_list.append(torch.tensor(cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)).unsqueeze(0))


                self.lable_list.append(position)

    def SaveData(self):
        self.left_eye_list = torch.stack(self.left_eye_list, 0)
        self.right_eye_list = torch.stack(self.right_eye_list, 0)
        self.lable_list = torch.stack(self.lable_list, 0)
        if len(self.pose_list) > 0:
            self.pose_list = torch.stack(self.pose_list, 0)
        torch.save({"left_eye":self.left_eye_list, "right_eye":self.right_eye_list, "lables":self.lable_list, "pose": self.pose_list}, self.dataset_dir + "dataset_partx.pt")

        self.webcam.release()