import warnings
import random
import numpy as np
import torch

# Setup seeds for reproducibility
def setup_seed():
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
