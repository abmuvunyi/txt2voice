import os
import random
import numpy as np
import time
import torch
from contextlib import contextmanager

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[{name}] {time.time() - t0:.2f}s")
