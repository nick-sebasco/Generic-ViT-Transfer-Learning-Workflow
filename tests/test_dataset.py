import pytest
import torch
import numpy as np
from src.analysis.dataset import ZarrDataset, load_features


def test_zarrdataset_initialization(dummy_features_targets):
    """Test that ZarrDataset initializes correctly."""
    features, targets, expected_length = dummy_features_targets
    dataset = ZarrDataset(features, targets)
    
    assert len(dataset) == expected_length  # Check length of the dataset
    x, t = dataset[0]
    assert x.shape == (3,)  # Check shape of the feature tensor
    assert t in [0, 1]  # Check target is a binary value


def test_zarrdataset_no_targets(dummy_features_targets):
    """Test that ZarrDataset handles the case without targets."""
    features, _, expected_length = dummy_features_targets
    dataset = ZarrDataset(features)
    
    assert len(dataset) == expected_length
    x, t = dataset[0]
    assert x.shape == (3,)  # Check shape of the feature tensor
    assert t is None  # No targets provided


def test_zarrdataset_indexing(dummy_features_targets):
    """Test that ZarrDataset returns the correct indexed samples."""
    features, targets, _ = dummy_features_targets
    dataset = ZarrDataset(features, targets)
    
    for idx in [0, 10, 99]:
        x, t = dataset[idx]
        assert torch.allclose(x, features[:, idx])  # Ensure the features match
        assert t == targets[idx]  # Ensure the targets match


def test_load_features_valid_positions(mock_zarr):
    """Test that load_features correctly filters valid positions."""
    features = load_features('dummy_path', '1')
    assert features.shape == (3, 10000)  # Flattened 100x100 grid to valid positions


def test_load_features_no_valid_positions(mock_zarr):
    """Test that load_features handles cases with no valid positions."""
    mock_zarr.open.return_value['ScanMask/1'][:] = 0  # No valid positions in mask
    features = load_features('dummy_path', '1')
    assert features.size == 0  # No valid features should be returned