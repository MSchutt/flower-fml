import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
from pathlib import Path

from diamondmodel.neural_network import MultipleRegression
from seed import setup_seed

class DiamondClient(fl.client.NumPyClient):
    def __init__(
            self,
            trainset: torchvision.datasets,
            device,
            num_features: int,
            log_file_path: Path,
            testset: torchvision.datasets
    ):
        self.device = device
        self.trainset = trainset
        self.num_features = num_features
        self.log_file_path = log_file_path
        self.testset = testset

    def set_parameters(self, parameters):
        """Loads a model and replaces it parameters with the ones received from the server."""
        model = MultipleRegression(self.num_features)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round (sent from the server)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Setup train and test loader
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(self.testset, batch_size=len(self.testset))

        results = utils.train(model, trainloader, testloader, epochs, self.device)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(self.trainset)

        # Evaluate global model parameters on the local test data and global test set and return results
        testloader = DataLoader(self.testset, batch_size=len(self.testset))
        trainloader = DataLoader(self.trainset, batch_size=batch_size)

        # Calculate both train and test loss
        loss, r2 = utils.test(model, trainloader, self.device)
        loss_test, r2_test = utils.test(model, testloader, self.device)

        # Additional write the loss into the log file (one for each client)
        with open(self.log_file_path, "a") as f:
            f.write(f"{loss},{r2},{loss_test},{r2_test}\n")

        return parameters_prime, num_examples_train, results

    def evaluate(
        self, parameters, config
    ):
        """Evaluate the model on the local test set."""
        model = self.set_parameters(parameters)
        testloader = DataLoader(self.testset, batch_size=len(self.testset))
        loss, r2 = utils.test(model, testloader, self.device)
        return float(loss), len(self.testset), {"r2": float(r2)}


def main() -> None:
    setup_seed()
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        required=False,
        help="Local Batch Size for the Clients",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=10,
        required=False,
        help="Local Batch Size for the Clients",
    )
    parser.add_argument(
        "--distribution",
        type=int,
        default=1,
        required=False,
        help="Should the data be distributed evenly or randomly",
    )
    parser.add_argument(
        "--runnumber",
        type=int,
        default=1,
        required=False,
        help="Run number (used for getting the correct logfile)",
    )
    args = parser.parse_args()

    # Load a subset of test to simulate the local data partition (every even client)
    trainset, testset, num_features = utils.load_partition(args.partition, args.clients, int(args.distribution) == 1)

    # All log files (r2, loss) are stored in the log folder with following folder structure
    # logfiles > [folder: runnumber; i.e. 1, 2, 3] > client_{{client-number}}.csv
    # These files can then be loaded and analyzed (to compare the local model performance)
    # Specify a log file
    log_file_path = Path("logfiles", str(int(args.runnumber)), f"client_{args.partition}.csv")
    # Ensure that the folder exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_path, "w") as f:
        f.write("loss,r2,loss_test,r2_test\n")

    # Start Flower client
    client = DiamondClient(trainset, device, num_features, log_file_path, testset)

    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    exit(0)

if __name__ == "__main__":
    main()
