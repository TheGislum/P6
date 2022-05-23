import cv2
import torch
import mouse
import ctypes
import keyboard
from gaze_model import annetV1, annetV2, annetV3
from torchvision import transforms
from argparse import ArgumentParser
from face_tracking import FaceTracking, FastFaceTracking
from eye_isolation import EyeIsolation, FastEyeIsolation
from pose_estimation import PoseEstimation
from calibration import Calibration

def main(model_version, model_colour, model_Path, use_fast_detection, calibrate, camera_resolution, verbose):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    user32 = ctypes.windll.user32
    screenWidth, screenHeight = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) # get monitor resolution

    img_transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    if calibrate:
        if verbose:
            print("Starting calibration")
        calibration = Calibration(model=model_Path, model_colour=model_colour, version=model_version,
                                    fast=use_fast_detection, camera_resolution=camera_resolution, verbose=verbose)
        offset_x, offset_y = calibration.Calibrate()
    else:
        if verbose:
            print("Skipping calibration")
        offset_x, offset_y = 0, 0
    
    if verbose:
        print(f"Calibration offset: x={offset_x}, y={offset_y}")

    if camera_resolution > 1080:
        camera_resolution = (10000, 10000)
    elif camera_resolution > 720:
        camera_resolution = (1920, 1080)
    elif camera_resolution > 480:
        camera_resolution = (1280, 720)
    else:
        camera_resolution = (720, 480)


    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])     #Set resolution of device
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
    ret, frame = webcam.read()

    if use_fast_detection:
        face = FastFaceTracking()
    else:
        face = FaceTracking()
    
    pose = PoseEstimation(frame=frame, fast=use_fast_detection)

    if verbose:
        print("Loading model...")

    if model_colour:
        model_channels = 6
    else:
        model_channels = 2

    if model_version == 1:
        net = annetV1(device=device, in_channels=model_channels)
        eye_res = (50, 30) #NOTE is this right?
    elif model_version == 2:
        net = annetV2(device=device, in_channels=model_channels)
        eye_res = (60, 36)
    elif model_version == 3:
        net = annetV3(device=device, in_channels=model_channels)
        eye_res = (60, 36)
    
    net.load_state_dict(torch.load(model_Path, map_location=device))
    net = net.eval()

    if verbose:
        print("Starting estimation (use 'c' to exit)")

    while True:
        # We get a new frame from the webcam
        ret, frame = webcam.read()
        if ret == True:
            if face.refresh(frame):
                if use_fast_detection:
                    eye_left = FastEyeIsolation(frame, face.landmarks, 0, eye_res).colour_frame
                    eye_right = FastEyeIsolation(frame, face.landmarks, 1, eye_res).colour_frame
                    
                else:
                    eye_left = EyeIsolation(frame, face.landmarks, 0, eye_res).colour_frame
                    eye_right = EyeIsolation(frame, face.landmarks, 1, eye_res).colour_frame

                if model_colour:
                    eye_left = torch.tensor(cv2.cvtColor(eye_left, cv2.COLOR_BGR2RGB), dtype=torch.float).permute(2,0,1)/255.0
                    eye_right = torch.tensor(cv2.cvtColor(eye_right, cv2.COLOR_BGR2RGB), dtype=torch.float).permute(2,0,1)/255.0
                    eye_left = img_transform(eye_left)
                    eye_right = img_transform(eye_right)
                else:
                    eye_left = torch.tensor(cv2.cvtColor(eye_left, cv2.COLOR_BGR2GRAY), dtype=torch.float).unsqueeze(0)/255.0
                    eye_right = torch.tensor(cv2.cvtColor(eye_right, cv2.COLOR_BGR2GRAY), dtype=torch.float).unsqueeze(0)/255.0
                
                input = torch.cat((eye_left, eye_right), 0).unsqueeze(0).to(device)

                if model_version == 3:
                    pose.refresh(frame, face.landmarks)

                    input = (input, pose.pose.unsqueeze(0).to(device))

                with torch.no_grad():
                    point = net(input).squeeze(0)
                
                x = ((point[0].item() + 1) * (screenWidth/2)) + offset_x
                y = ((1 - point[1].item()) * (screenHeight/2)) + offset_y

                if verbose:
                    print(f"model estimates: X={x} Y={y}")

                mouse.move(x, y)
            

        if keyboard.is_pressed('c') == True:
            break
    
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_version", type=int, default=2, help="which model version to use (default: 2)")
    parser.add_argument("--model_colour", action="store_true", default=False, help="use colour images (default: False)")
    parser.add_argument("--model_Path", type=str, default="./garage/03.pth", help="gaze model path (default: 03)")
    parser.add_argument("-fast", "--use_fast_detection", action="store_true", default=False, help="use fast face/landmark detection (default: False)")
    parser.add_argument("-c", "--calibrate_off", action="store_false", default=True, help="run calibration (default: True)")
    parser.add_argument("-cr", "--camera_resolution", type=int, default=10000, help="camera resolution input to model (default: max)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    main(args.model_version, args.model_colour, args.model_Path, args.use_fast_detection, args.calibrate_off, args.camera_resolution, args.verbose)

#https://github.com/vardanagarwal/Proctoring-AI
#https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/