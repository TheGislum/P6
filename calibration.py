import cv2
from numpy import eye
import torch
import mouse
import ctypes
import keyboard
import os
import glob
from gaze_model import annet
from torchvision import transforms
from face_tracking import FaceTracking
from eye_isolation import EyeIsolation
from pose_estimation import PoseEstimation
from data_collector import DataCollector

class Calibration:
    def __init__(self, model = './garage/both_epoch_1000_0.009294.pth', channels = 6):
        self.datacollector = DataCollector("calibration")
        self.model = model
        self.channels = channels

    def Calibrate(self):
        dataset_dir = "./calibration_dataset/"
        for file in glob.glob(dataset_dir):
            os.remove(file)
        self.Collect()
        file = "dataset_partx.pt"
        dataset = torch.load(os.path.join(dataset_dir, file))
        left_eye_images = dataset["left_eye"]
        right_eye_images = dataset["right_eye"]
        label_list = dataset["lables"]
        predicted_points = []
        user32 = ctypes.windll.user32
        screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # get monitor resolution

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = annet(device=device, in_channels=self.channels)
        net.load_state_dict(torch.load(self.model, map_location=device))
        net = net.eval()

        for i in range(len(left_eye_images)):
            point = self.GetPredictedPoint(net, device, left_eye_images[i], right_eye_images[i], screenWidth, screenHeight)
            predicted_points.append(point)

        offset_x, offset_y = self.GetOffsets(label_list, predicted_points)
        return offset_x, offset_y

    def Collect(self):
        self.datacollector.RunCollection()

    def GetPredictedPoint(self, net, device, left_eye_image, right_eye_image, screenWidth, screenHeight):
        img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eye_left = img_transform(left_eye_image / 255.0)
        eye_right = img_transform(right_eye_image / 255.0)

        # input = eye_left
        input = torch.cat((eye_left, eye_right), 0)
        with torch.no_grad():
            point = net(input.unsqueeze(0).to(device)).squeeze(0)

        x = ((point[0].item() + 1) * (screenWidth / 2))
        y = ((1 - point[1].item()) * (screenHeight / 2))
        return [x, y]

    def GetOffsets(self, label_list, predicted_points):
        offset_x = 0
        offset_y = 0
        for i in range(len(label_list)):
            user32 = ctypes.windll.user32
            screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # get monitor resolution
            label_x = (label_list[i][0].item() + 1) * (screenWidth / 2)
            label_y = (1 - label_list[i][1].item()) * (screenHeight / 2)
            offset_x = offset_x + (label_x - predicted_points[i][0])
            offset_y = offset_y + (label_y - predicted_points[i][1])
        offset_x = offset_x / len(label_list)
        offset_y = offset_y / len(label_list)

        return offset_x, offset_y

calibration = Calibration()
print(calibration.Calibrate())