import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from pathlib import Path
from torch import nn, optim

from diamondmodel.dataset import generate_dataloaders
from diamondmodel.evaluate import evaluate_model
from diamondmodel.learn import train_model
from diamondmodel.loader import load_data
from diamondmodel.neural_network import DiamondNN

from seed import setup_seed

# Parameters for the centralized NN
BATCH_SIZE = 32
EPOCHS = 55
LEARNING_RATE = 0.001

# Helper function to calculate MAE
def mean_absolute_error(y_test, y_pred):
    return np.mean(np.abs(y_pred - y_test))

# Helper function to print scores to stdout
def calculate_and_print_scores(name, model, X_train, y_train, X_test, y_test):
    r2sq = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mean_absolute_error_test = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2sq_train = model.score(X_train, y_train)
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mean_absolute_error_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)

    print(f'''
        Model {name} Scores:
        *** TRAINSET ***
        R2: {r2sq_train}
        MSE: {mse_train}
        RMSE: {rmse_train}
        Mean Absolute Err: {mean_absolute_error_test}
        ****************

        *** TESTSET ***
        Model {name} Scores:
        R2: {r2sq}
        MSE: {mse}
        RMSE: {rmse}
        Mean Absolute Err: {mean_absolute_error_train}
        ****************
    ''')

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Airbnb Data Prediction")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


# Helper function to brute force the best parameters
def brute_force_model(X_train, y_train):
    rfr_local = RandomForestRegressor()
    # Possible values
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator=rfr_local, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    print(rf_random.best_estimator_)
    print(rf_random.best_score_)
    return rf_random.best_estimator_


def nn_results(X_train, y_train, X_test, y_test):
    # Load the data
    train_loader, test_loader, train_dataset, test_dataset = generate_dataloaders(X_train, y_train, X_test, y_test,
                                                                                  BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create and train the model
    num_features = len(X_train.columns)
    model = DiamondNN(num_features)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_stats = train_model(model, criterion, optimizer, train_loader, test_loader, EPOCHS, device)

    # Evaluate
    evaluate_model(model, X_test, y_test, device)

    # Show training progress
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
        columns={"index": "epochs"})
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.show()

# Centralized reference model
if __name__ == '__main__':
    setup_seed()

    pth = Path('data/diamonds.csv')
    (X_train, y_train), (X_test, y_test) = load_data(pth)

    # Uncomment this to brute force the best parameters (takes a while!)
    # brute_force_model(X_train, y_train)
    print('Fitting RandomForestRegressor')
    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    calculate_and_print_scores('Random Forest', rfr, X_train, y_train, X_test, y_test)
    print('Finished fitting RandomForestRegressor')

    # Linear regression
    print('Fitting LinearRegression')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    calculate_and_print_scores('LinearRegression', lr, X_train, y_train, X_test, y_test)
    print('Finished fitting LinearRegression')

    # NN
    print('Fitting NN')
    nn_results(X_train, y_train, X_test, y_test)
    print('Finished fitting NN')

