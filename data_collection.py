import cv2
import mouse
import ctypes
import keyboard
import pandas as pd
from face_tracking import FaceTracking
from pose_estimation import PoseEstimation
from eye_isolation import EyeIsolation

dataset_dir = "./eye_dataset/"
dataset = pd.read_csv(dataset_dir + "./eye_dataset.csv")

webcam = cv2.VideoCapture(0)
ret, img = webcam.read()

face = FaceTracking()
pose = PoseEstimation(img)

step = 100
user32 = ctypes.windll.user32
screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) # get monitor resolution
#screenWidth, screenHeight = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79) # get multi monitor resolution (all monitors)
stop = False
j = 0

mouse.move(step, step)

while keyboard.is_pressed('s') != True:
    i = 0


for i in range(step, screenWidth-step, step):
    if stop:
        break
    if i != step:
        mouse.move(i, step, duration=1.5)
    for j in range(step, screenHeight-step, step):
        mouse.move(i, j, duration=(step/120)*0.5) # steps_per_second=120
        screenX, screenY = mouse.get_position()
        x = (screenX/(screenWidth/2))-1
        y = 1-(screenY/(screenHeight/2))

        if keyboard.is_pressed('c') == True:
            stop = True
        if stop:
            break

        for p in range(5):
            # We get a new frame from the webcam
            ret, frame = webcam.read()
            if ret == True:
                if face.refresh(frame):
                    pose.refresh(frame, face.landmarks)
                    left_eye = EyeIsolation(frame, pose.face_landmarks, 0, (50, 30))
                    right_eye = EyeIsolation(frame, pose.face_landmarks, 1, (50, 30))
                    
                    if ((left_eye.blinking + right_eye.blinking) / 2) < 4.2:
                        row, col = dataset.shape # maby row + x?
                        left_eye_path = "left_eye" + str(row) + ".png"
                        right_eye_path = "right_eye" + str(row) + ".png"

                        cv2.imwrite(dataset_dir + left_eye_path, left_eye.colour_frame)
                        cv2.imwrite(dataset_dir + right_eye_path, right_eye.colour_frame)

                        dataset = pd.concat([dataset, pd.DataFrame({'left_eye': [left_eye_path], 'right_eye': [right_eye_path], 'x': [x], 'y': [y]}, columns=dataset.columns)])
   
webcam.release()
dataset.to_csv(dataset_dir + "eye_dataset.csv", index = False, header = True, mode = 'w')

#https://github.com/boppreh/mouse


# import keyboard
# import mouse
# import ctypes

# step = 150
# user32 = ctypes.windll.user32
# screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) # get monitor resolution
# stop = False
# j = 0

# mouse.move(step, step)

# while keyboard.is_pressed('s') != True:
#     i = 0

# for i in range(step, screenWidth-step, step):
#     if stop:
#         break
#     if i != step:
#         mouse.move(i, step, duration=1.5)
#     for j in range(step, screenHeight-step, step):
#         mouse.move(i, j, duration=(step/120)*0.5) # steps_per_second=120
#         screenX, screenY = mouse.get_position()
#         if keyboard.is_pressed('c') == True:
#             stop = True
#         if stop:
#             break