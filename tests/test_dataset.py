import pytest
import torch
import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from src.analysis.dataset import ZarrDataset
from zarr.errors import PathNotFoundError  


def test_zarrdataset_initialization_with_metadata(
    dummy_image_ids, dummy_zarr_files, dummy_metadata
):
    """Test initialization with valid inputs and metadata."""
    dataset = ZarrDataset(
        image_ids=dummy_image_ids,
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
        metadata_path=dummy_metadata,
        target_column='target',
        class_order=[0, 1, 2],
        ordinal=False,
    )
    assert len(dataset) == len(dummy_image_ids)
    assert dataset.metadata_path == dummy_metadata
    assert dataset.target_column == 'target'
    assert dataset.class_order == [0, 1, 2]
    assert not dataset.ordinal

    expected_targets = {
        'image_1': np.array([1, 0, 0]),
        'image_2': np.array([0, 1, 0]),
        'image_3': np.array([0, 0, 1]),
    }

    for image_id in expected_targets:
        np.testing.assert_array_equal(
            dataset.image_id_to_target[image_id],
            expected_targets[image_id]
        )


def test_zarrdataset_initialization_without_metadata(
    dummy_image_ids, dummy_zarr_files
):
    """Test initialization without metadata."""
    dataset = ZarrDataset(
        image_ids=dummy_image_ids,
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
    )
    assert len(dataset) == len(dummy_image_ids)
    assert dataset.metadata_path is None
    assert dataset.target_column is None
    assert dataset.image_id_to_target == {}


def test_zarrdataset_len(dummy_image_ids, dummy_zarr_files):
    """Test __len__ method."""
    dataset = ZarrDataset(
        image_ids=dummy_image_ids,
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
    )
    assert len(dataset) == 3


def test_zarrdataset_getitem_with_metadata(
    dummy_image_ids, dummy_zarr_files, dummy_metadata
):
    """Test __getitem__ method with metadata and one-hot encoding."""
    dataset = ZarrDataset(
        image_ids=dummy_image_ids,
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
        metadata_path=dummy_metadata,
        target_column='target',
        class_order=[0, 1, 2],
        ordinal=False,
    )
    for idx in range(len(dataset)):
        feature_tensor, target_tensor = dataset[idx]
        assert feature_tensor.shape == torch.Size([768])
        assert target_tensor.shape == torch.Size([3])
        assert torch.sum(target_tensor) == 1  # One-hot encoded


def test_zarrdataset_getitem_with_ordinal_encoding(
    dummy_image_ids, dummy_zarr_files, dummy_metadata
):
    """Test __getitem__ method with metadata and ordinal encoding."""
    dataset = ZarrDataset(
        image_ids=dummy_image_ids,
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
        metadata_path=dummy_metadata,
        target_column='target',
        class_order=[0, 1, 2],
        ordinal=True,
    )
    expected_targets = [
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([1, 1, 1]),
    ]
    for idx in range(len(dataset)):
        feature_tensor, target_tensor = dataset[idx]
        assert feature_tensor.shape == torch.Size([768])
        np.testing.assert_array_equal(target_tensor.numpy(), expected_targets[idx])


def test_zarrdataset_getitem_without_metadata(
    dummy_image_ids, dummy_zarr_files
):
    """Test __getitem__ method without metadata."""
    dataset = ZarrDataset(
        image_ids=dummy_image_ids,
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
    )
    for idx in range(len(dataset)):
        feature_tensor = dataset[idx]
        assert feature_tensor.shape == torch.Size([768])


def test_zarrdataset_missing_target_in_metadata(
    dummy_image_ids, dummy_zarr_files, dummy_metadata
):
    """Test behavior when image_id is missing from metadata."""
    # Remove an image_id from metadata
    df = pd.read_csv(dummy_metadata)
    df = df[df['image_id'] != 'image_3']
    metadata_path = os.path.join(os.path.dirname(dummy_metadata), 'incomplete_metadata.csv')
    df.to_csv(metadata_path, index=False)

    dataset = ZarrDataset(
        image_ids=dummy_image_ids,
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
        metadata_path=metadata_path,
        target_column='target',
        class_order=[0, 1, 2],
    )
    with pytest.raises(KeyError):
        dataset[2]


def test_zarrdataset_empty_image_ids(dummy_feature_dir):
    """Test behavior with empty image_ids list."""
    dataset = ZarrDataset(
        image_ids=[],
        feature_zarr_dir=str(dummy_feature_dir),
        agg_type='mean',
        resolution='1.0',
    )
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        dataset[0]


def test_zarrdataset_single_image_id(
    dummy_zarr_files, dummy_metadata
):
    """Test with a single image_id."""
    dataset = ZarrDataset(
        image_ids=['image_1'],
        feature_zarr_dir=str(dummy_zarr_files),
        agg_type='mean',
        resolution='1.0',
        metadata_path=dummy_metadata,
        target_column='target',
        class_order=[0, 1, 2],
    )
    assert len(dataset) == 1
    feature_tensor, target_tensor = dataset[0]
    assert feature_tensor.shape == torch.Size([768])
    np.testing.assert_array_equal(target_tensor.numpy(), [1, 0, 0])


def test_zarrdataset_invalid_class_order(
    dummy_image_ids, dummy_zarr_files, dummy_metadata
):
    """Test behavior when class_order does not match targets."""
    with pytest.raises(KeyError):
        ZarrDataset(
            image_ids=dummy_image_ids,
            feature_zarr_dir=str(dummy_zarr_files),
            agg_type='mean',
            resolution='1.0',
            metadata_path=dummy_metadata,
            target_column='target',
            class_order=[3, 4, 5],
        )

def test_zarrdataset_missing_zarr_file(
    dummy_image_ids, dummy_feature_dir, dummy_metadata
):
    """Test behavior when zarr file is missing."""
    dataset = ZarrDataset(
        image_ids=['image_missing'],
        feature_zarr_dir=str(dummy_feature_dir),
        agg_type='mean',
        resolution='1.0',
        metadata_path=dummy_metadata,
        target_column='target',
    )
    with pytest.raises(PathNotFoundError):  # Expect the correct exception
        dataset[0]