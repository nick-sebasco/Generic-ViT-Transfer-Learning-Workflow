import torch
import pytest

@pytest.fixture
def dummy_input():
    """Fixture to create a dummy input tensor."""
    return torch.randn(32, 512)  # 32 samples, 512-dimensional input