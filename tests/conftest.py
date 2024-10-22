import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytest
import zarr
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
def dummy_image_ids():
    return ['image_1', 'image_2', 'image_3']


@pytest.fixture
def dummy_feature_dir(tmp_path):
    # Create a temporary directory for feature zarrs
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    return feature_dir


@pytest.fixture
def dummy_zarr_files(dummy_image_ids, dummy_feature_dir):
    # Create dummy zarr files for each image_id
    for image_id in dummy_image_ids:
        zarr_path = dummy_feature_dir / f"ViT_features_{image_id}.zarr"
        zarr_dataset = zarr.open(str(zarr_path), mode='w')
        # Create dummy features
        zarr_dataset.create_dataset(
            "SlideLevelFeatures/mean/1.0", data=np.random.rand(1, 768)
        )
    return dummy_feature_dir


@pytest.fixture
def dummy_metadata(tmp_path, dummy_image_ids):
    # Create a dummy metadata CSV file
    metadata_path = tmp_path / "metadata.csv"
    data = {
        'image_id': dummy_image_ids,
        'target': [0, 1, 2]
    }
    df = pd.DataFrame(data)
    df.to_csv(metadata_path, index=False)
    return str(metadata_path)

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
