import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from diamondmodel.dataset import generate_dataloaders
from diamondmodel.evaluate import evaluate_model
from diamondmodel.learn import train_model
from diamondmodel.loader import load_data
from pathlib import Path
import seaborn as sns

from diamondmodel.neural_network import MultipleRegression, MultipleRegressionSmall

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

def main():
    pth = Path('./data/diamonds.csv')
    # Load the data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data(pth)
    train_loader, val_loader, test_loader, train_dataset, valid_dataset, test_dataset = generate_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create and train the model
    num_features = len(X_train.columns)
    model = MultipleRegression(num_features)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_stats = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS, device)


    # Evaluate
    evaluate_model(model, X_test, y_test, device)

    # Show training progress
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.show()





if __name__ == '__main__':
    main()
