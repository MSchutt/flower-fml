from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse

import numpy as np
from time import time
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader

import flwr as fl
import torch

import utils

import warnings
import random
from diamondmodel.dataset import generate_dataloaders
from diamondmodel.loader import load_data
from diamondmodel.neural_network import MultipleRegressionSmall, MultipleRegression
from seed import seed_worker

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
g = torch.Generator()
g.manual_seed(42)


# Controls the amount of local batches used for training
def get_fit_config(batch_size: int, local_epochs: int):
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "batch_size": batch_size,
            "local_epochs": local_epochs,
        }
        return config
    return fit_config


def get_evaluate_fn(model: torch.nn.Module, testset):
    """Return an evaluation function for server-side evaluation."""

    test_loader = DataLoader(testset, batch_size=16, worker_init_fn=seed_worker)
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, r2 = utils.test(model, test_loader)
        return loss, {"r2": r2}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    start = time()
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data()
    train_loader, val_loader, test_loader, train_dataset, valid_dataset, test_dataset = generate_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, 32)

    # Create the model
    model = MultipleRegressionSmall(len(X_train.columns))
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32,
        required=False,
        help="Local Batch Size for the Clients",
    )
    parser.add_argument(
        "--localepochs",
        type=int,
        default=32,
        required=False,
        help="Local Epochs for the Clients",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        required=False,
        help="Number of Clients that are started",
    )
    parser.add_argument(
        "--numrounds",
        type=int,
        default=10,
        required=False,
        help="Number of Rounds for each client",
    )
    args = parser.parse_args()

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=args.clients,
        # Server side metrics
        evaluate_fn=get_evaluate_fn(model, test_dataset),
        on_fit_config_fn=get_fit_config(batch_size=args.batchsize, local_epochs=args.localepochs),
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.numrounds),
        strategy=strategy,
    )

    # Use the real testing set (never seen before for the clients) only on the server
    # Clients optimize locally with validation set
    testing_X = torch.from_numpy(X_test.values).float()
    testing_y = torch.from_numpy(y_test.values).float()

    with torch.no_grad():
        predictions = model(testing_X)
        predictions = predictions.cpu().numpy()
        # calculate statistics
        r2 = r2_score(testing_y, predictions)
        mse = mean_squared_error(testing_y, predictions)
        rmse = np.sqrt(mse)

        print(f'R2 Score: {r2}')
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')

    result = {
        "local_epochs": args.localepochs,
        "local_batch_size": args.batchsize,
        "clients": args.clients,
        "rounds": args.numrounds
    }

    # Write to results file
    with open("results.csv", "a") as a:
        a.write(f'{result["local_epochs"]},{result["local_batch_size"]},{result["clients"]},{result["rounds"]},{r2},{mse},{rmse},{time()-start}\n')

    exit(0)


if __name__ == "__main__":
    main()

