import torch
import numpy as np
from torch import nn, optim
from gaze_model import annet
from eye_dataset import eyeDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def main():
    EPOCHS = 1000
    BATCH_SIZE = 64
    WEIGHT_DECAY = 0.001
    LEARNING_RATE = 0.0001
    PRINT_EVERY = 64
    SAVE = './garage/both'
    dataset_dir = './eye_dataset/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = annet(device=device, in_channels=6)
    #net.load_state_dict(torch.load('./garage/metr_epoch_1000_0.005517.pth', map_location=device))

    img_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset = eyeDataset(dataset_dir + "dataset2.csv", dataset_dir + "dataset2", img_transform, True, True)
    
    train_split = 0.9
    train_set, test_set = random_split(dataset, [int(len(dataset)*train_split), int(len(dataset)-(int(len(dataset)*train_split)))])

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    train_loss = []

    print("start training...", flush=True)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            # print statistics
            running_loss += loss.item()
            if i % PRINT_EVERY == 0:    # print every PRINT_EVERY mini-batches
                print(f'[{epoch + 1}, {i + 1:3d}] loss: {np.mean(train_loss[-PRINT_EVERY:]):.6f}')
                running_loss = 0.0

    torch.save(net.state_dict(), SAVE + "_epoch_" + str(0+EPOCHS) + "_" + str(round(np.mean(train_loss), 6)) + ".pth")

    print("Finished Training - loss: {:.4f}".format(np.mean(train_loss)))
    print("start testing...", flush=True)
    test_loss = []
    
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss.append(loss.item())

            # print statistics
            running_loss += loss.item()
            if i % PRINT_EVERY == 0:    # print every PRINT_EVERY mini-batches
                print(f'[{epoch + 1}, {i + 1:3d}] loss: {np.mean(train_loss[-PRINT_EVERY:]):.6f}')
                running_loss = 0.0

    print("Finished Testing - loss: {:.4f}".format(np.mean(test_loss)))

if __name__ == "__main__":
    main()

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network