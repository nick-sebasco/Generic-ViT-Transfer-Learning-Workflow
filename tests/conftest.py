import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

# test_models
# --------------------
@pytest.fixture
def dummy_input():
    """Fixture to create a dummy input tensor."""
    return torch.randn(32, 512)  # 32 samples, 512-dimensional input


# test_dataset
# --------------------
@pytest.fixture
def dummy_features_targets():
    """Fixture to create dummy feature and target arrays."""
    length = 100
    features = torch.randn(3, length)  # 3 features, {length} samples
    targets = torch.randint(0, 2, (length,))  # {length} binary targets
    return features, targets, length


@pytest.fixture
def mock_zarr(monkeypatch):
    """Fixture to mock Zarr data access."""
    zarr_mock = MagicMock()
    features = np.random.rand(3, 100, 100)  # 3 features, 100x100 grid
    scan_mask = np.ones((100, 100))  # All positions valid (mask == 1)
    zarr_mock.open.return_value = {'Features/1': features, 'ScanMask/1': scan_mask}
    monkeypatch.setattr('src.analysis.dataset.zarr', zarr_mock)
    return zarr_mock


# test_training
# --------------------