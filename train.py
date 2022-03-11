import cv2
import torch
import numpy as np
from gaze_model import model

EPOCHS = 5
BATCH_SIZE = 64
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.001
PRINT_EVERY = 1000
SAVE = './garage/metr'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO load a dataset with the right formatting
image_set = torch.load

trainloader = torch.utils.data.DataLoader(image_set, BATCH_SIZE, shuffle=True, num_workers=2)

optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, WEIGHT_DECAY)
criterion = torch.nn.MSELoss()
train_loss = []

print("start training...", flush=True)

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.preprocess_image(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss)

        # print statistics
        running_loss += loss.item()
        if i % PRINT_EVERY == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_EVERY:.3f}')
            running_loss = 0.0

torch.save(model.state_dict(), SAVE + "_epoch_" + str(EPOCHS) + "_" + str(round(np.mean(train_loss), 6)) + ".pth")

print('Finished Training')

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network