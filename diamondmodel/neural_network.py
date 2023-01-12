from torch import nn

# Neural Network
class MultipleRegression(nn.Module):
    def __init__(self, num_features: int):
        super(MultipleRegression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, inputs):
        return self.layers(inputs)

    def predict(self, test_inputs):
        return self.forward(test_inputs)

# Neural Network small
class MultipleRegressionSmall(nn.Module):
    def __init__(self, num_features: int):
        super(MultipleRegressionSmall, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, inputs):
        return self.layers(inputs)

    def predict(self, test_inputs):
        return self.forward(test_inputs)