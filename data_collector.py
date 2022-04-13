import cv2
import mouse
import ctypes
import keyboard
import torch
from face_tracking import FaceTracking
from eye_isolation import EyeIsolation

class DataCollector:
    def __init__(self, calibration = None):
        if calibration is None:
            self.dataset_dir = "./eye_dataset/"
            self.stepX = 80
            self.stepY = 80
            self.x_duration = (80/120)*0.5
        else:
            self.dataset_dir = "./calibration_dataset/"
            self.stepX = ctypes.windll.user32.GetSystemMetrics(0) - 80
            self.stepY = ctypes.windll.user32.GetSystemMetrics(1) - 40
            self.x_duration = 1.5
        self.collected = 0
        self.left_eye_list = []
        self.right_eye_list = []
        self.lable_list = []
        self.startX = 40
        self.startY = 20
        self.face = FaceTracking()
        self.webcam = cv2.VideoCapture(0)

    def RunCollection(self):
        ret, img = self.webcam.read()

        user32 = ctypes.windll.user32
        screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) # get monitor resolution
        #screenWidth, screenHeight = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79) # get multi monitor resolution (all monitors)
        stop = False
        j = 0

        mouse.move(self.startX, self.startY)

        while keyboard.is_pressed('s') != True:
            i = 0

        for i in range(self.startX, (screenWidth - self.startX) + self.stepX, self.stepX):
            if stop:
                break
            if i != self.startX:
                mouse.move(i, self.startY, duration=1.5)
            for j in range(self.startY, (screenHeight - self.startY) + self.stepY, self.stepY):
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
                left_eye = EyeIsolation(frame, self.face.landmarks, 0, (50, 30)).colour_frame
                right_eye = EyeIsolation(frame, self.face.landmarks, 1, (50, 30)).colour_frame

                self.left_eye_list.append(torch.tensor(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)).permute(2,0,1))
                self.right_eye_list.append(torch.tensor(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)).permute(2,0,1))

                self.lable_list.append(position)

    def SaveData(self):
        self.left_eye_list = torch.stack(self.left_eye_list, 0)
        self.right_eye_list = torch.stack(self.right_eye_list, 0)
        self.lable_list = torch.stack(self.lable_list, 0)
        torch.save({"left_eye":self.left_eye_list, "right_eye":self.right_eye_list, "lables":self.lable_list}, self.dataset_dir + "dataset_partx.pt")

        self.webcam.release()