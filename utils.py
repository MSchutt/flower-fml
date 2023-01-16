import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import r2_score, mean_squared_error
from torch import nn, optim
from torchvision.datasets import CIFAR10

import warnings

from diamondmodel.dataset import generate_dataloaders, DiamondDataset
from diamondmodel.loader import load_data

warnings.filterwarnings("ignore")

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_partition(idx: int, tp: int, enable_small_dataset: bool = False):
    total_partitions = tp + 1
    """Load 1/10th of the training and test data to simulate a partition."""
    (X_train, y_train), (X_valid, y_valid), _ = load_data()
    train_dataset = DiamondDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
    valid_dataset = DiamondDataset(torch.from_numpy(X_valid.values).float(), torch.from_numpy(y_valid.values).float())

    num_examples = {
        "trainset": len(train_dataset),
        "valset": len(valid_dataset)
    }

    n_train = int(num_examples["trainset"] / total_partitions)
    n_test = int(num_examples["valset"] / total_partitions)

    # If the parameter is enabled, we only take 25% of the available data
    if enable_small_dataset:
        n_train = int(n_train * .25)
        n_test = int(n_test * .25)

    train_partition = torch.utils.data.Subset(train_dataset, range(idx * n_train, (idx + 1) * n_train))
    valid_partition = torch.utils.data.Subset(valid_dataset, range(idx * n_test, (idx + 1) * n_test))
    return train_partition, valid_partition, len(X_valid.columns)


def train(net, trainloader, valloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for X_train_batch, y_train_batch in trainloader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            train_loss = criterion(net(X_train_batch), y_train_batch.unsqueeze(1))
            train_loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    train_loss, train_r2 = test(net, trainloader)
    val_loss, val_r2 = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_r2": train_r2,
        "val_loss": val_loss,
        "val_r2": val_r2
    }
    return results


def test(net, testloader, device: str = "cpu"):
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    total, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (X_test_batch, y_test_batch) in enumerate(testloader):
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            predictions = net(X_test_batch.to(device))
            predictions = predictions.cpu().numpy()
            # calculate statistics
            total += y_test_batch.size(0)
            r2 = r2_score(y_test_batch, predictions)
            # mse = mean_squared_error(y_test_batch, predictions)
            # rmse = np.sqrt(mse)
    if total > 0:
        loss /= total
    net.to("cpu")  # move model back to CPU
    return loss, r2

def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
