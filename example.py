import cv2
from face_tracking import FaceTracking
from pose_estimation import PoseEstimation

webcam = cv2.VideoCapture(0)
ret, img = webcam.read()

face = FaceTracking()
pose = PoseEstimation(img)

while True:
    # We get a new frame from the webcam
    ret, frame = webcam.read()
    if ret == True:
        face.refresh(frame)
        face.draw_face_squares(frame)
        face.draw_landmarks(frame)
        pose.refresh(frame, face.landmarks)
        pose.draw_facing(frame)
        pose.write_position_on_frame(frame)
    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()