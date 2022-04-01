import cv2
from numpy import eye
import torch
import mouse
import ctypes
import keyboard
from gaze_model import annet
from torchvision import transforms
from face_tracking import FaceTracking
from eye_isolation import EyeIsolation
from pose_estimation import PoseEstimation

webcam = cv2.VideoCapture(0)
ret, img = webcam.read()

face = FaceTracking()
pose = PoseEstimation(img)

user32 = ctypes.windll.user32
screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) # get monitor resolution

img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = annet(device=device, in_channels=3)
net.load_state_dict(torch.load('./garage/left_epoch_2000_0.005099.pth', map_location=device))
net = net.eval()

while True:
    # We get a new frame from the webcam
    ret, frame = webcam.read()
    if ret == True:
        if face.refresh(frame):
            pose.refresh(frame, face.landmarks)
            eye_left = EyeIsolation(frame, pose.face_landmarks, 0, (50, 30)).colour_frame
            eye_left = torch.tensor(cv2.cvtColor(eye_left, cv2.COLOR_BGR2RGB), dtype=torch.float).permute(2,0,1)/255.0
            eye_left = img_transform(eye_left)
            
            # eye_right = EyeIsolation(frame, pose.face_landmarks, 1, (50, 30)).colour_frame
            # eye_right = torch.tensor(cv2.cvtColor(eye_right, cv2.COLOR_BGR2RGB), dtype=torch.float).permute(2,0,1)/255.0
            # eye_right = img_transform(eye_right)
            
            input = eye_left
            # input = torch.cat((eye_left, eye_right), 0)
            with torch.no_grad():
                point = net(input.unsqueeze(0).to(device)).squeeze(0)
            
            x = ((point[0].item() + 1) * (screenWidth/2))
            y = ((1 - point[1].item()) * (screenHeight/2))
            mouse.move(x, y)
        

    if keyboard.is_pressed('c') == True:
            break
   
webcam.release()
cv2.destroyAllWindows()

#https://github.com/vardanagarwal/Proctoring-AI
#https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/