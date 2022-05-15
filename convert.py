import os
import re
import torch
from PIL import Image
import numpy as np
import pandas as pd
from zipfile import ZipFile

#dataset location
dataset_dir = './UT Multi-view Gaze Dataset'
result_dir = './eye_dataset'

# use test or synth part of the dataset?
use_dataset_part = "test"

def decode_headpose_txt(file_str):
    #Split string into lines
    split_str = file_str.split('\n')
    scientific_notation = r"[+\-]?(?=\.\d|\d)(?:0|[1-9]\d*)?(?:\.\d+)?(?:(?<=\d)(?:[eE][+\-]?\d+))?"
    #HeadPose list:
    headpose_list = split_str[1]
    #Get data from HeadPose list using regex:
    headpose_list = re.findall(r"[-+]?\d*\.\d+|\d+", headpose_list)
    headpose_list = [float(i) for i in headpose_list]

    #HeadPose matrix:
    headpose_matrix = split_str[2:5]
    #Get data from HeadPose matrix using regex:
    headpose_matrix_list = []
    for line in headpose_matrix:
        #Match scientific notation:
        line_list = re.findall(scientific_notation, line)
        #Convert to float:
        line_list = [float(i) for i in line_list]
        #Append to list:
        headpose_matrix_list.append(line_list)

    #Features matrix:
    features_matrix = split_str[6:12]
    #Get data from Features matrix using regex:
    features_matrix_list = []
    for line in features_matrix:
        #Match scientific notation:
        line_list = re.findall(scientific_notation, line)
        #Convert to float:
        line_list = [float(i) for i in line_list]
        #Append to list:
        features_matrix_list.append(line_list)
    
    return [headpose_list, headpose_matrix_list, features_matrix_list]

def decode_cparams_txt(file_str):
    #Split string into lines
    split_str = file_str.split('\n')
    scientific_notation = r"[+\-]?(?=\.\d|\d)(?:0|[1-9]\d*)?(?:\.\d+)?(?:(?<=\d)(?:[eE][+\-]?\d+))?"
    
    #Features matrix:
    features_matrix = split_str[1:4]
    #Get data from CONTOUR matrix using regex:
    features_matrix_list = []
    for line in features_matrix:
        #Match scientific notation:
        line_list = re.findall(scientific_notation, line)
        #Convert to float:
        line_list = [float(i) for i in line_list]
        #Append to list:
        features_matrix_list.append(line_list)
    
    return features_matrix_list

datapoints = 8

if use_dataset_part == "test":
    use_dataset_part = "test/"
    datapoints = 8
elif use_dataset_part == "synt":
    use_dataset_part = "synth/"
    datapoints = 144
else:
    use_dataset_part = "test/"
    datapoints = 8

left_eye_list = []
right_eye_list = []
lable_list = []
true_lable_list = []
head_pose = []
cparams = []

for file in os.listdir(dataset_dir):
    if '.zip' in file:
        with ZipFile(os.path.join(dataset_dir, file), 'r') as zip:
            files = zip.namelist()
            for file in files:
                if 's' in file and len(file) < 5:
                    gazedata = pd.read_csv(zip.open(file + 'raw/gazedata.csv'), header=0).iloc[:,1:]
                    for lable in range(len(gazedata)):
                        x = (gazedata.iloc[lable,0]/(473.8/2))-1
                        y = 1-(gazedata.iloc[lable,1]/(296.1/2))
                        for i in range(datapoints):
                            lable_list.append(torch.tensor([x, y], dtype=torch.float))
                    
                    left_eye_folder = []
                    right_eye_folder = []
                    left_eye_data_file = []
                    right_eye_data_file = []
                    head_pose_data_file = []
                    cparams_file = []

                    for test_folder in files:
                        if file + use_dataset_part in test_folder and '.zip' in test_folder and '.csv' not in test_folder: # zip folder contaning eye images
                            if 'right' in test_folder:
                                left_eye_folder.append(test_folder)
                            else:
                                right_eye_folder.append(test_folder)
                        elif file + use_dataset_part in test_folder and '.zip' not in test_folder and '.csv' in test_folder: # csv file with lables
                            if 'right' in test_folder:
                                left_eye_data_file.append(test_folder)
                            else:
                                right_eye_data_file.append(test_folder)
                        elif file + 'raw/img' in test_folder and 'cparams' in test_folder and '.txt' in test_folder: # cparams data
                            cparams_file.append(test_folder)
                        elif file + 'raw/' in test_folder and 'headpose.txt' in test_folder: # headpose data
                            head_pose_data_file.append(test_folder)

                    
                    for i in range(len(left_eye_folder)): # extract left and right eye images
                        with ZipFile(zip.open(left_eye_folder[i]), 'r') as data:
                            for img in data.namelist():
                                image = Image.open(data.open(img))
                                left_eye = np.array(image)
                                left_eye_list.append(torch.tensor(left_eye).unsqueeze(0))

                        with ZipFile(zip.open(right_eye_folder[i]), 'r') as data:
                            for img in data.namelist():
                                image = Image.open(data.open(img))
                                right_eye = np.array(image)
                                right_eye_list.append(torch.tensor(right_eye).unsqueeze(0))

                    left_data = []
                    right_data = []
                    for i in range(len(left_eye_data_file)):
                        left_data.append(pd.read_csv(zip.open(left_eye_data_file[i]), header=None).iloc[:,:9])
                        right_data.append(pd.read_csv(zip.open(right_eye_data_file[i]), header=None).iloc[:,:9])
                    
                    left_data = pd.concat(left_data, axis=0, ignore_index=True).values.tolist()
                    right_data = pd.concat(right_data, axis=0, ignore_index=True).values.tolist()

                    for i in range(len(left_data)):
                        true_lable_list.append([left_data[i], right_data[i]])

                    for pose_data in head_pose_data_file:
                        data = zip.open(pose_data)
                        data = decode_headpose_txt(data.read().decode('UTF-8'))
                        for i in range(datapoints):
                            head_pose.append(data)

                    for cparams_data in cparams_file:
                        data = zip.open(cparams_data)
                        data = decode_cparams_txt(data.read().decode('UTF-8'))
                        cparams.append(data)
                        


left_eye_list = torch.stack(left_eye_list, 0)
right_eye_list = torch.stack(right_eye_list, 0)
lable_list = torch.stack(lable_list, 0)
torch.save({"left_eye":left_eye_list, "right_eye":right_eye_list, "lables":lable_list, "true_lable":true_lable_list, "head_pose":head_pose, "cparams":cparams},
            result_dir + "/dataset_part_test.pt")