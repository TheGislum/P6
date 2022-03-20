import cv2
from face_tracking import FaceTracking
from pose_estimation import PoseEstimation
from eye_isolation import EyeIsolation

webcam = cv2.VideoCapture(0)
ret, img = webcam.read()

face = FaceTracking()
pose = PoseEstimation(img)

while True:
    # We get a new frame from the webcam
    ret, frame = webcam.read()
    if ret == True:
        if face.refresh(frame):
            pose.refresh(frame, face.landmarks)
            eye_left = EyeIsolation(frame, pose.face_landmarks, 0, (50, 30))
            eye_right = EyeIsolation(frame, pose.face_landmarks, 1, (50, 30))
            cv2.imshow("eye_left", eye_left.colour_frame)
            #cv2.putText(frame, str(((eye_left.blinking + eye_right.blinking) / 2)), (45, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, str((eye_left.colour_frame.shape)), (45, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            
        face.draw_face_squares(frame)
        face.draw_landmarks(frame)

        pose.draw_facing(frame)
        pose.write_position_on_frame(frame)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()

#https://github.com/vardanagarwal/Proctoring-AI
#https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/