import os
import cv2
import math
import torch
import numpy as np
from torch.utils.data import Dataset

class eyeDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_left_eye=True, use_right_eye=True, pose=True):
        self.transform = transform
        self.use_left_eye = use_left_eye
        self.use_right_eye = use_right_eye
        self.pose = pose
        self.left_eye_images = None
        self.right_eye_images = None
        self.lables = None
        self.cparams = None
        self.head_pose = None

        for file in os.listdir(root_dir):
            if 'dataset_part' in file and '.pt' in file:
                dataset = torch.load(os.path.join(root_dir, file))
                if self.lables is None:
                    self.left_eye_images = dataset["left_eye"]
                    self.right_eye_images = dataset["right_eye"]
                    self.lables = dataset["lables"]
                    if self.pose:
                        self.cparams = dataset["cparams"]
                        self.head_pose = dataset["head_pose"]
                else:
                    self.left_eye_images = torch.cat((self.left_eye_images, dataset["left_eye"]), 0)
                    self.right_eye_images = torch.cat((self.right_eye_images, dataset["right_eye"]), 0)
                    self.lables = torch.cat((self.lables, dataset["lables"]), 0)
                    if self.pose:
                        self.cparams.extend(dataset["cparams"])
                        self.head_pose.extend(dataset["head_pose"])

    def _rotationMatrixToEulerAngles(self, R) :

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, index):
        if self.use_left_eye:
            left_eye_img = self.left_eye_images[index]/255
        if self.use_right_eye:
            right_eye_img = self.right_eye_images[index]/255

        if self.transform is not None:
            if self.use_left_eye:
                left_eye_img = self.transform(left_eye_img)
            if self.use_right_eye:
                right_eye_img = self.transform(right_eye_img)

        if self.use_left_eye and self.use_right_eye:
            img = torch.cat((left_eye_img, right_eye_img),0)
        elif self.use_right_eye:
            img = right_eye_img
        elif self.use_left_eye:
            img = left_eye_img
        else:
            img = None

        lable = self.lables[index]

        if not self.pose:
            return (img, lable)

        else:
            camres = (1280, 1024)
            projMatrix = np.array(self.cparams[index])
            head_pose = self.head_pose[index]
            head_rotation = np.array(head_pose[1])
            features = np.array(head_pose[2])
            point1 = np.array(features[0])
            point4 = np.array(features[3])
            origin = np.append(np.divide(np.add(point1, point4), 2), 1)

            xyz = projMatrix @ origin
            z = xyz[2]
            xyz /= z
            xyz[0] = camres[0]
            xyz[1] = camres[1]
            xyz[2] = z / 10000

            euler = self._rotationMatrixToEulerAngles(head_rotation @ cv2.decomposeProjectionMatrix(projMatrix)[1])

            pose = torch.tensor(np.hstack((xyz, euler)))

            return (img, lable, pose)