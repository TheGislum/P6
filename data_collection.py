import cv2
import mouse
import ctypes
import keyboard
import torch
from face_tracking import FaceTracking
from eye_isolation import EyeIsolation

dataset_dir = "./eye_dataset/"

webcam = cv2.VideoCapture(0)
ret, img = webcam.read()

face = FaceTracking()

step = 80
user32 = ctypes.windll.user32
screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) # get monitor resolution
#screenWidth, screenHeight = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79) # get multi monitor resolution (all monitors)
stop = False
j = 0
startX, startY = 40, 20

left_eye_list = []
right_eye_list = []
lable_list = []

mouse.move(startX, startY)

while keyboard.is_pressed('s') != True:
    i = 0

for i in range(startX, (screenWidth - startX) + step, step):
    if stop:
        break
    if i != startX:
        mouse.move(i, startY, duration=1.5)
    for j in range(startY, (screenHeight - startY) + step, step):
        mouse.move(i, j, duration=(step/120)*0.5) # steps_per_second=120
        screenX, screenY = mouse.get_position()
        x = (screenX/(screenWidth/2))-1
        y = 1-(screenY/(screenHeight/2))
        posistion = torch.tensor([x, y], dtype=torch.float)

        if keyboard.is_pressed('c') == True:
            stop = True
        if stop:
            break

        for p in range(5):
            # We get a new frame from the webcam
            ret, frame = webcam.read()
            if ret == True:
                if face.refresh(frame):
                    left_eye = EyeIsolation(frame, face.landmarks, 0, (50, 30)).colour_frame
                    right_eye = EyeIsolation(frame, face.landmarks, 1, (50, 30)).colour_frame

                    left_eye_list.append(torch.tensor(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)).permute(2,0,1))
                    right_eye_list.append(torch.tensor(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)).permute(2,0,1))

                    lable_list.append(posistion)

left_eye_list = torch.stack(left_eye_list, 0)
right_eye_list = torch.stack(right_eye_list, 0)
lable_list = torch.stack(lable_list, 0)
torch.save({"left_eye":left_eye_list, "right_eye":right_eye_list, "lables":lable_list}, dataset_dir + "dataset_partx.pt")          
   
webcam.release()

#https://github.com/boppreh/mouse