import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
@pytest.fixture
def small_dummy_train_validate_data():
    """Fixture to create a small dummy dataset for testing."""
    inputs = torch.randn(10, 5)  # 10 samples, 5 features
    targets = torch.randint(0, 2, (10,))  # 10 binary targets
    return inputs, targets


@pytest.fixture
def simple_train_validate_model():
    """Fixture to create a simple model for testing."""
    return nn.Linear(5, 1)  # Linear model with 5 inputs and 1 output


@pytest.fixture
def simple_train_validate_optimizer(simple_train_validate_model):
    """Fixture to create an optimizer for the model."""
    return optim.SGD(simple_train_validate_model.parameters(), lr=0.01)


@pytest.fixture
def criterion():
    """Fixture to create a loss function."""
    return nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits loss
