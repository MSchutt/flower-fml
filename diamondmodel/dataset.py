import torch
from torch.utils.data import DataLoader

from seed import seed_worker


# PyTorch Dataset
class DiamondDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# Helper function to create the PyTorch Dataloaders
def generate_dataloaders(X_train, y_train, X_test, y_test, batch):
    train_dataset = DiamondDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
    test_dataset = DiamondDataset(torch.from_numpy(X_test.values).float(), torch.from_numpy(y_test.values).float())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=True, worker_init_fn=seed_worker)

    return train_loader, test_loader, train_dataset, test_dataset