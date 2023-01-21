from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse

import numpy as np
from time import time
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader

import flwr as fl
import torch
from pathlib import Path

import utils

import warnings
import random
from diamondmodel.dataset import generate_dataloaders
from diamondmodel.loader import load_data
from diamondmodel.neural_network import MultipleRegression, MultipleRegression
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


def get_evaluate_fn(model: torch.nn.Module, batch_size, testset, log_path: Path, params):
    """Return an evaluation function for server-side evaluation."""

    test_loader = DataLoader(testset, batch_size=batch_size, worker_init_fn=seed_worker)

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
    (X_train, y_train), (X_test, y_test) = load_data()
    train_loader, test_loader, train_dataset, test_dataset = generate_dataloaders(X_train, y_train, X_test, y_test, 32)

    # Create the model
    model = MultipleRegression(len(X_train.columns))
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
    parser.add_argument(
        "--distribution",
        type=int,
        default=1,
        required=False,
        help="Skewed distribution (1) or uniform distribution (0)",
    )
    parser.add_argument(
        "--clientSamplingRatio",
        type=float,
        default=1.0,
        required=False,
        help="Specify client sampling percentage",
    )
    parser.add_argument(
        "--runnumber",
        type=int,
        default=1,
        required=False,
        help="Specify the run number",
    )
    args = parser.parse_args()

    # All log files (r2, loss) are stored in the log folder with following folder structure
    # logfiles > [folder: runnumber; i.e. 1, 2, 3] > client_{{client-number}}.csv
    # These files can then be loaded and analyzed (to compare the local models performance)
    # Specify a log file
    log_file_path = Path("logfiles", str(int(args.runnumber)), "server.csv")
    # Ensure that the folder exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "local_epochs": args.localepochs,
        "local_batch_size": args.batchsize,
        "clients": args.clients,
        "rounds": args.numrounds,
        "distribution": args.distribution,
        "runnumber": args.runnumber,
        "samplingRatio": args.clientSamplingRatio
    }

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # percentage of total client that are required to respond from the clients
        # before the server starts a new federated round
        fraction_fit=args.clientSamplingRatio,
        fraction_evaluate=1.0,
        # min_fit_clients : int, optional
        min_available_clients=args.clients,
        # Server side metrics
        evaluate_fn=get_evaluate_fn(model, args.batchsize, test_dataset, log_file_path, params),
        on_fit_config_fn=get_fit_config(batch_size=args.batchsize, local_epochs=args.localepochs),
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    test = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.numrounds),
        strategy=strategy,
    )

    print(test.losses_centralized)
    print(test.metrics_centralized)

    with open(log_file_path, "a") as f:
        for (_, r2) in test.metrics_centralized["r2"]:
            f.write(f"{r2}\n")

    # Store final r2
    r2 = test.metrics_centralized["r2"][-1][1]

    # Store server logs
    # Write to results file
    with open("results.csv", "a") as a:
        a.write(
            f'{params["runnumber"]},{params["samplingRatio"]},{params["local_epochs"]},{params["local_batch_size"]},{params["clients"]},{params["rounds"]},{r2},{time() - start},{params["distribution"]}\n')

    exit(0)


if __name__ == "__main__":
    main()
