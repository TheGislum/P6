import os
import torch
from torch.utils.data import Dataset

class eyeDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_left_eye=True, use_right_eye=True):
        self.transform = transform
        self.left_eye = use_left_eye
        self.right_eye = use_right_eye
        self.eyes = None
        self.lables = None

        for file in os.listdir(root_dir):
            if 'dataset_part' in file and '.pt' in file:
                dataset = torch.load(os.path.join(root_dir, file))
                if self.eyes is None:
                    self.eyes = dataset["eyes"]
                    self.lables = dataset["lables"]
                else:
                    self.eyes = torch.cat((self.eyes, dataset["eyes"]), 0)
                    self.lables = torch.cat((self.lables, dataset["lables"]), 0)


    def __len__(self):
        return len(self.eyes)

    def __getitem__(self, index):
        concatenated_eyes = self.eyes[index]/255
        left_eye_img, right_eye_img = concatenated_eyes.chunk(2,0)

        if self.transform is not None:
            if self.left_eye:
                left_eye_img = self.transform(left_eye_img)
            if self.right_eye:
                right_eye_img = self.transform(right_eye_img)

        if self.left_eye and self.right_eye:
            img = torch.cat((left_eye_img, right_eye_img),0)
        elif self.right_eye:
            img = right_eye_img
        elif self.left_eye:
            img = left_eye_img
        else:
            img = None

        lable = self.lables[index]

        return (img, lable)