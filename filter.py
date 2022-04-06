import os
import numpy as np
import torch
import cv2

dataset_dir = "./eye_dataset/"
file = "dataset_partx.pt"

dataset = torch.load(os.path.join(dataset_dir, file))
left_eye_images = dataset["left_eye"]
right_eye_images = dataset["right_eye"]
label_list = dataset["lables"]

left_eye_imgs_to_keep = []
right_eye_imgs_to_keep = []
labels_to_keep = []
save = True

for i in range(len(left_eye_images)):
    cv2.imshow("Eyes", cv2.resize(np.concatenate((cv2.cvtColor(left_eye_images[i].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR), cv2.cvtColor(right_eye_images[i].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)), axis = 1), (400, 120)))
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