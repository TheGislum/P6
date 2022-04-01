import os
import torch
from torch.utils.data import Dataset

class eyeDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_left_eye=True, use_right_eye=True):
        self.transform = transform
        self.use_left_eye = use_left_eye
        self.use_right_eye = use_right_eye
        self.left_eye_images = None
        self.right_eye_images = None
        self.lables = None

        for file in os.listdir(root_dir):
            if 'dataset_part' in file and '.pt' in file:
                dataset = torch.load(os.path.join(root_dir, file))
                if self.lables is None:
                    self.left_eye_images = dataset["left_eye"]
                    self.right_eye_images = dataset["right_eye"]
                    self.lables = dataset["lables"]
                else:
                    self.left_eye_images = torch.cat((self.left_eye_images, dataset["left_eye"]), 0)
                    self.right_eye_images = torch.cat((self.right_eye_images, dataset["right_eye"]), 0)
                    self.lables = torch.cat((self.lables, dataset["lables"]), 0)


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

        return (img, lable)