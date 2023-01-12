import warnings
import random
import numpy as np
import torch

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
g = torch.Generator()
g.manual_seed(42)

def seed_worker(worker_id):
    np.random.seed(42)
    random.seed(42)