from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from gaze_model import annetV2
from eye_dataset import eyeDataset

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Loss


def get_data_loaders(train_batch_size, val_batch_size, data_split, dataset_dir):
    img_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    dataset = eyeDataset(dataset_dir, img_transform, True, True)

    train_set, test_set = random_split(dataset, [int(len(dataset)*data_split), int(len(dataset)-(int(len(dataset)*data_split)))])

    train_loader = DataLoader(train_set, train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(test_set, val_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def run(train_batch_size, val_batch_size, data_split, dataset_dir, epochs, lr, weight_decay, log_interval, save_dir):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, data_split, dataset_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = annetV2(device=device, in_channels=2)

    model.to(device)  # Move model before creating optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    criterion = nn.MSELoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    val_metrics = {"MSE": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=f"ITERATION - loss: {0:.6f}")

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_start(engine):
        tqdm.write(f"Epoch {engine.state.epoch} started")

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.desc = f"ITERATION - loss: {engine.state.output:.6f}"
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        tqdm.write(f"Evaluating traning data on model...")
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_MSE = metrics["MSE"]
        tqdm.write(f"Training Results - Epoch: {engine.state.epoch} Avg loss: {avg_MSE:.6f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_time(engine):
        tqdm.write(f"{trainer.last_event_name.name} took { trainer.state.times[trainer.last_event_name.name]} seconds")

        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.COMPLETED)
    def log_validation_results(engine):
        tqdm.write(f"Evaluating validation data on model...")
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_MSE = metrics["MSE"]
        tqdm.write(f"Validation Results - Epoch: {engine.state.epoch} Avg loss: {avg_MSE:.6f}")

    @trainer.on(Events.COMPLETED)
    def log_time(engine):
        tqdm.write(f"{trainer.last_event_name.name} took { trainer.state.times[trainer.last_event_name.name]} seconds")

    @trainer.on(Events.COMPLETED)
    def save_model(engine):
        avg_MSE = evaluator.state.metrics["MSE"]
        model_name = f"gazeModel_epoch_{engine.state.epoch}_loss_{avg_MSE:.6f}.pth"
        tqdm.write(f"{trainer.last_event_name.name}: saving model as: {model_name}")
        torch.save(model, save_dir + model_name)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=128, help="input batch size for validation (default: 128)")
    parser.add_argument("--data_split", type=float, default=0.9, help="training/validation dataset split (default: 0.9)")
    parser.add_argument("--dataset_dir", type=str, default='./eye_dataset/', help="dataset directory")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight_decay (default: 0.0)")
    parser.add_argument("--log_interval", type=int, default=10, help="how many batches to wait before logging training status")
    parser.add_argument("--save_dir", type=str, default='./garage/', help="directory to save the model in (default: ./garage/)")

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.data_split, args.dataset_dir, args.epochs, args.lr, args.weight_decay, args.log_interval, args.save_dir)