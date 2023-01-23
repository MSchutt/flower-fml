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


# Returns a partition given the model's index and the total number of clients
def get_partition(dataset, partition_idx, total_partition):
    total_dataset = len(dataset)

    n = int(total_dataset / total_partition)
    partition = torch.utils.data.Subset(dataset, range(partition_idx * n, (partition_idx + 1) * n))

    return partition

# Returns start and end range of a given partition
def get_partition_range(n_total, total_clients, idx, uneven=False):
    """Get the range of indices for the partition."""
    normal_partition = int(n_total / total_clients)

    # Calculate an uneven partition (10% and 190% of the desired partition size)
    if uneven:
        # Smaller batch -> only 10% of the partition data
        small_batch = int(normal_partition * .1)
        # Larger batch -> 190% of the partition data
        large_batch = int(normal_partition * 1.9)
        start = 0
        for i in range(idx):
            if i % 2 == 1:
                start += small_batch
            else:
                start += large_batch
        end = start + (small_batch if idx % 2 == 1 else large_batch)
        return int(start), int(end)



    return idx * normal_partition, (idx + 1) * normal_partition

# Load a partition of the dataset
def load_partition(idx: int, total_partitions: int, enable_small_dataset: bool = False):
    (X_train, y_train), (X_test, y_test) = load_data()
    train_dataset = DiamondDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
    testset = DiamondDataset(torch.from_numpy(X_test.values).float(), torch.from_numpy(y_test.values).float())

    # Get the ranges of the partition (including uneven distributions)
    train_start, train_end = get_partition_range(len(train_dataset) - 1, total_partitions, idx, uneven=enable_small_dataset)

    # Get the partition from the pytorch dataset
    train_partition = torch.utils.data.Subset(train_dataset, range(train_start, (train_end if train_end < len(train_dataset) else len(train_dataset))))
    return train_partition, testset, len(X_train.columns)

def train(net, trainloader, testloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
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
    test_loss, test_r2 = test(net, testloader)

    results = {
        "train_loss": float(train_loss),
        "train_r2": train_r2,
        "test_loss": float(test_loss),
        "test_r2": test_r2,
    }
    return results


def test(net, testloader, device: str = "cpu"):
    """Validate the network on the entire test set."""
    net.to(device)  # move model to GPU if available
    total, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (X_test_batch, y_test_batch) in enumerate(testloader):
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            predictions = net(X_test_batch.to(device))
            # get loss
            loss += nn.MSELoss()(predictions, y_test_batch.unsqueeze(1).to(device))
            predictions = predictions.cpu().numpy()
            # calculate statistics
            total += y_test_batch.size(0)
            r2 = r2_score(y_test_batch, predictions)
    net.to("cpu")  # move model back to CPU
    return float(loss), r2

def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

