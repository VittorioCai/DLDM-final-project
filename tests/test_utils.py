import numpy as np
import torch

from shortcut_learning.utils import choose_device, set_seed


def test_set_seed_reproduces_numpy_and_torch_draws():
    set_seed(7)
    numpy_first = np.random.random(4)
    torch_first = torch.rand(4)

    set_seed(7)
    numpy_second = np.random.random(4)
    torch_second = torch.rand(4)

    np.testing.assert_array_equal(numpy_first, numpy_second)
    torch.testing.assert_close(torch_first, torch_second)


def test_choose_device_auto_returns_supported_device():
    assert choose_device("auto").type in {"cpu", "mps", "cuda"}


def test_choose_device_honors_explicit_cpu():
    assert choose_device("cpu").type == "cpu"
