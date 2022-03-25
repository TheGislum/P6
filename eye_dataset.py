import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class eyeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, left_eye=True, right_eye=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.left_eye = left_eye
        self.right_eye = right_eye

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if self.left_eye:
            left_eye_img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
            left_eye_img = torch.tensor(cv2.cvtColor(cv2.imread(left_eye_img_path), cv2.COLOR_BGR2RGB), dtype=torch.float).permute(2,0,1)/255.0

        if self.right_eye:
            right_eye_img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
            right_eye_img = torch.tensor(cv2.cvtColor(cv2.imread(right_eye_img_path), cv2.COLOR_BGR2RGB), dtype=torch.float).permute(2,0,1)/255.0

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

        lable = torch.tensor([self.annotations.iloc[index, 2], self.annotations.iloc[index, 3]], dtype=torch.float) #TODO how to combine?

        return (img, lable)