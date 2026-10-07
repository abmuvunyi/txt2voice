import random
import numpy as np
import torch
from src.utils import seed_everything, timer


def test_seed_everything():
    seed_everything(42)
    assert random.randint(0, 100) == 81
    assert np.random.randint(0, 100) == 51
    if torch.cuda.is_available():
        assert torch.backends.cudnn.deterministic is True


def test_timer(capsys):
    with timer("test_timer"):
        pass
    captured = capsys.readouterr()
    assert "[test_timer]" in captured.out
