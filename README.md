# Federated Machine Learning Experiment using Flower and neural networks
* This project consists of a federated learning experiment using Flower and neural networks. The experiment is based on the [Flower tutorial](https://flower.dev/docs/tutorial.html) and the [Flower documentation](https://flower.dev/docs/quickstart.html).

## Folder Structure
* `data` - contains the data used for the experiment
* `diamondmodel` - contains the neural network and various helper function (data loading, data preprocessing, evaluation, training) used for the experiment
* `analysis` - contains the analysis of the experiment (plots, results, etc.)
* `logfiles` - contains the log files generated by the experiment and follow this naming schema:     
  * `logfiles > [folder: runnumber; i.e. 1, 2, 3] > client_{{client-number}}.csv/server.csv` and store the r2 value for the testset after each time the `fit` function is called (after each federated round for each client and server)
* `tests` - contains the unit tests for the partition helper function
* .gitignore - contains the files and folders that should be ignored by git
* centralized.py - contains the code for the centralized experiment (baseline models, linear regression, random forest, neural network)
* client.py - contains the code for the client
* poetry.lock - contains the dependencies and their versions
* pyproject.toml - contains the dependencies and their versions
* README.md - this file
* results.csv - contains the results of the experiment
* server.py - contains the code for the server
* seed.py - helper function for reproducibility to set the seed for the experiment
* start_experiment.py - starting point for the simulation
* utils.py - helper functions for the experiment

## Getting started & Prerequisites
* Install [Poetry](https://python-poetry.org/docs/#installation)
* Install the dependencies by running `poetry install` in the root directory of the project
* Run `poetry shell` to activate the virtual environment
* Run `python centralized.py` to run the baseline models (linear regression, random forest, neural network)
* Run `python start_experiment.py` to start the experiment (this will take a while)
