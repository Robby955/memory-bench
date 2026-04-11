"""Shared fixtures for memory-bench tests.

Ensures test isolation by managing random state, preventing cross-test
contamination from leaked torch/numpy/python random state.
"""

import random
import torch
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate_random_state():
    """Save and restore all random state around each test.

    Without this, test ordering affects results because torch's global
    RNG state leaks between tests. Any test that calls torch.randn()
    or similar without setting a seed first would get different values
    depending on what ran before it.
    """
    # Save state
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    cuda_states = []
    if torch.cuda.is_available():
        cuda_states = [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]

    yield

    # Restore state
    torch.random.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    if torch.cuda.is_available():
        for d, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, d)
