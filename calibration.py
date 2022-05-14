import torch
import ctypes
import os
import glob
from gaze_model import annetV1, annetV2, annetV3
from torchvision import transforms
from data_collector import DataCollector

class Calibration:
    def __init__(self, model = './garage/03.pth', model_colour = False, version = 2, fast=False, verbose=False):
        self.model = model
        self.model_colour = model_colour
        if model_colour:
            self.channels = 6
        else:
            self.channels = 2
        self.version = version
        self.verbose = verbose
        self.left_eye_images = None
        self.right_eye_images = None
        self.pose = None
        self.label_list = None
        self.predicted_points = []
        self.datacollector = DataCollector(calibration="calibration", fast=fast, model_colour=self.model_colour, version=version)
        
        user32 = ctypes.windll.user32
        self.screenWidth, self.screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # get monitor resolution

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.version == 1:
            self.net = annetV1(device=self.device, in_channels=self.channels)
        elif self.version == 2:
            self.net = annetV2(device=self.device, in_channels=self.channels)
        elif self.version == 3:
            self.net = annetV3(device=self.device, in_channels=self.channels)

    def Calibrate(self):
        dataset_dir = "./calibration_dataset/"
        for file in glob.glob(dataset_dir):
            os.remove(file)
        self.Collect()
        file = "dataset_partx.pt"
        dataset = torch.load(os.path.join(dataset_dir, file))
        self.left_eye_images = dataset["left_eye"]
        self.right_eye_images = dataset["right_eye"]
        self.pose = dataset["pose"]
        self.label_list = dataset["lables"]
        
        self.net.load_state_dict(torch.load(self.model, map_location=self.device))
        self.net = self.net.eval()

        for i in range(len(self.left_eye_images)):
            point = self.GetPredictedPoint(i)
            self.predicted_points.append(point)

        offset_x, offset_y = self.GetOffsets()
        return offset_x, offset_y

    def Collect(self):
        self.datacollector.RunCollection()

    def GetPredictedPoint(self, index):
        img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.model_colour:
            eye_left = img_transform(self.left_eye_image[index] / 255.0)
            eye_right = img_transform(self.right_eye_image[index] / 255.0)
        else: 
            eye_left = self.left_eye_image[index] / 255.0
            eye_right = self.right_eye_image[index] / 255.0

        input = torch.cat((eye_left, eye_right), 0).unsqueeze(0).to(self.device)

        if self.version == 3:
            input = (input, self.pose[index].unsqueeze(0).to(self.device))

        with torch.no_grad():
            point = self.net(input).squeeze(0)

        x = ((point[0].item() + 1) * (self.screenWidth / 2))
        y = ((1 - point[1].item()) * (self.screenHeight / 2))
        
        return [x, y]

    def GetOffsets(self):
        offset_x = 0
        offset_y = 0
        for i in range(len(self.label_list)):
            label_x = (self.label_list[i][0].item() + 1) * (self.screenWidth / 2)
            label_y = (1 - self.label_list[i][1].item()) * (self.screenHeight / 2)
            offset_x = offset_x + (label_x - self.predicted_points[i][0])
            offset_y = offset_y + (label_y - self.predicted_points[i][1])
        offset_x = offset_x / len(self.label_list)
        offset_y = offset_y / len(self.label_list)

        return offset_x, offset_y

if __name__ == "__main__":
    calibration = Calibration()
    print(calibration.Calibrate())