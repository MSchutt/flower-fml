import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict

from diamondmodel.neural_network import MultipleRegression, MultipleRegressionSmall
from seed import seed_worker
g = torch.Generator()
g.manual_seed(42)

class DiamondClient(fl.client.NumPyClient):
    def __init__(
            self,
            trainset: torchvision.datasets,
            valset: torchvision.datasets,
            device,
            num_features: int
    ):
        self.device = device
        self.trainset = trainset
        self.valset = valset
        self.num_features = num_features

    def set_parameters(self, parameters):
        """Loads a model and replaces it parameters with the ones received from the server."""
        model = MultipleRegressionSmall(self.num_features)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held validation set."""

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round (sent from the server)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Setup train and validation loader
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(self.valset, batch_size=batch_size, worker_init_fn=seed_worker)

        results = utils.train(model, train_loader, val_loader, epochs, self.device)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(self.trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Evaluate global model parameters on the local validation data and return results
        valloader = DataLoader(self.valset, batch_size=16, worker_init_fn=seed_worker)

        loss, r2 = utils.test(model, valloader, self.device)
        return float(loss), len(self.valset), {"r2": float(r2)}


def main() -> None:
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
    args = parser.parse_args()

    enable_small_dataset = args.distribution == 1

    # Load a subset of test to simulate the local data partition
    trainset, valset, num_features = utils.load_partition(args.partition, args.clients)

    # Start Flower client
    client = DiamondClient(trainset, valset, device, num_features)

    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    exit(0)

if __name__ == "__main__":
    main()
