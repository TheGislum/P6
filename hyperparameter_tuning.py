from functools import partial
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest import ConcurrencyLimiter
from gaze_model import annetV3
from eye_dataset import eyeDataset


def load_data(data_dir="./eye_dataset/"):
    img_transform = transforms.Compose([ # bw transform
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])
        
    dataset = eyeDataset(data_dir, img_transform, True, True)

    return dataset


def train_cifar(config, checkpoint_dir='./checkpoints', data_dir=None):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net = annetV3(device, in_channels=2)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 200 == 199:  # print every 2000 mini-batches
                #print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                tune.report(loss=(running_loss / epoch_steps))
                running_loss = 0.0
                epoch_steps = 0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                labels = labels.to(device)

                outputs = net(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    data_dir = os.path.abspath("./eye_dataset/")
    load_data(data_dir)
    config = {
        "lr": tune.loguniform(1e-8, 1.5e-6),
        "weight_decay": 0.0, #tune.loguniform(1e-4, 5e-1),
        "batch_size": 64 #tune.choice([16, 32, 64, 128])
    }
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=max_num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_num_epochs,
        )
    bohb_search = ConcurrencyLimiter(TuneBOHB(), max_concurrent=4)
    reporter = CLIReporter(
        parameter_columns=["lr", "weight_decay", "batch_size"],
        metric_columns=["loss", "training_iteration"]
        )
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        name="gaze_model_V3_BOHB_lr_wd0",
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        progress_reporter=reporter,
        metric="loss",
        mode="min",
        resume="AUTO",
        )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=16, max_num_epochs=20, gpus_per_trial=0.5)


# from functools import partial
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import random_split
# import torchvision.transforms as transforms
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
# from gaze_model import annetV2
# from eye_dataset import eyeDataset


# def load_data(data_dir="./eye_dataset/"):
#     img_transform = transforms.Compose([ # bw transform
#             transforms.ColorJitter(brightness=0.3, contrast=0.3),
#         ])
        
#     dataset = eyeDataset(data_dir, img_transform, True, True)

#     return dataset


# def train_cifar(config, checkpoint_dir='./checkpoints', data_dir=None):

#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#     net = annetV2(device, in_channels=2)

#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(
#             os.path.join(checkpoint_dir, "checkpoint"))
#         net.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)

#     trainset = load_data(data_dir)

#     test_abs = int(len(trainset) * 0.8)
#     train_subset, val_subset = random_split(
#         trainset, [test_abs, len(trainset) - test_abs])

#     trainloader = torch.utils.data.DataLoader(
#         train_subset,
#         batch_size=int(config["batch_size"]),
#         shuffle=True,
#         num_workers=8)
#     valloader = torch.utils.data.DataLoader(
#         val_subset,
#         batch_size=int(config["batch_size"]),
#         shuffle=True,
#         num_workers=8)

#     for epoch in range(10):  # loop over the dataset multiple times
#         running_loss = 0.0
#         epoch_steps = 0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             epoch_steps += 1
#             if i % 200 == 199:  # print every 2000 mini-batches
#                 #print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
#                 tune.report(loss=(running_loss / epoch_steps))
#                 running_loss = 0.0
#                 epoch_steps = 0

#         # Validation loss
#         val_loss = 0.0
#         val_steps = 0
#         for i, data in enumerate(valloader, 0):
#             with torch.no_grad():
#                 inputs, labels = data
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 outputs = net(inputs)

#                 loss = criterion(outputs, labels)
#                 val_loss += loss.cpu().numpy()
#                 val_steps += 1

#         with tune.checkpoint_dir(epoch) as checkpoint_dir:
#             path = os.path.join(checkpoint_dir, "checkpoint")
#             torch.save((net.state_dict(), optimizer.state_dict()), path)

#         tune.report(loss=(val_loss / val_steps))
#     print("Finished Training")


# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
#     data_dir = os.path.abspath("./eye_dataset/")
#     load_data(data_dir)
#     config = {
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "weight_decay": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([16, 32, 64, 128])
#     }
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2
#         )
#     reporter = CLIReporter(
#         parameter_columns=["lr", "weight_decay", "batch_size"],
#         metric_columns=["loss", "training_iteration"]
#         )
#     result = tune.run(
#         partial(train_cifar, data_dir=data_dir),
#         name="gaze_model_ASHA",
#         resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter
#         )

#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))


# if __name__ == "__main__":
#     # You can change the number of GPUs per trial here:
#     main(num_samples=50, max_num_epochs=10, gpus_per_trial=0.5)