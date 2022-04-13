import os
import numpy as np
import torch
import cv2

dataset_dir = "./eye_dataset/"
file = "dataset__part_martin_0.pt"

dataset = torch.load(os.path.join(dataset_dir, file))
left_eye_images = dataset["left_eye"]
right_eye_images = dataset["right_eye"]
label_list = dataset["lables"]


left_eye_images = left_eye_images.permute(0, 2, 3, 1).numpy()
right_eye_images = right_eye_images.permute(0, 2, 3, 1).numpy()
label_list = label_list.numpy()

left_eye_imgs_to_keep = []
right_eye_imgs_to_keep = []
labels_to_keep = []
save = True
length = len(left_eye_images)

for i in range(length):
    # frame = cv2.resize(np.concatenate((left_eye_images[i], right_eye_images[i]), axis = 1), (400, 120)) # BW
    frame = cv2.resize(np.concatenate((cv2.cvtColor(left_eye_images[i], cv2.COLOR_RGB2BGR), cv2.cvtColor(right_eye_images[i], cv2.COLOR_RGB2BGR)), axis = 1), (400, 120)) # RGB
    cv2.putText(frame, f"{i}/{length}", (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"{label_list[i]}", (200, 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
    cv2.imshow("Eyes", frame)
    key = cv2.waitKey(0)
    if key == 107: # if key == 'k'
        left_eye_imgs_to_keep.append(left_eye_images[i])
        right_eye_imgs_to_keep.append(right_eye_images[i])
        labels_to_keep.append(label_list[i])
    elif key == 27: # if key == 'ESC'
        save = False
        break
cv2.destroyAllWindows()

if save:
    left_eye_images = torch.stack(left_eye_imgs_to_keep,0)
    right_eye_images = torch.stack(right_eye_imgs_to_keep,0)
    label_list = torch.stack(labels_to_keep,0)
    torch.save({"left_eye":left_eye_images, "right_eye":right_eye_images, "lables":label_list}, dataset_dir + "filtered_" + file) 