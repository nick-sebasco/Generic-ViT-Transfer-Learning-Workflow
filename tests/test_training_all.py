import zarr
import numpy as np
import torch 
import pytest
import os
import shutil
from typing import List
from src.analysis.training import run_training


def clean_up(directories: List[str]) -> None:
    """
    Creates a dummy Zarr file with random features and a scan mask for testing.

    Parameters:
    ----------
    path : List[str]
    List of directories to remove

    Returns:
    -------
    None
    """
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)


def create_dummy_zarr(path: str, num_samples=200, input_dim=5, resolution='1'):
    """
    Creates a dummy Zarr file with random features and a scan mask for testing.

    Parameters:
    ----------
    path : str
        Path where the Zarr file will be saved.
    num_samples : int
        Number of samples in the dummy dataset.
    input_dim : int
        Number of input features per sample.
    resolution : str
        The resolution key used in the Zarr store.
    """
    # Create a Zarr group at the specified path
    root = zarr.open(path, mode='w')
    
    # Create a random feature array and a scan mask with valid positions
    features = np.random.randn(input_dim, num_samples).astype(np.float32)
    scan_mask = np.ones((num_samples,), dtype=bool)  # All positions are valid (1 means valid)
    
    # Create Zarr arrays to store the data
    root.create_dataset(f'Features/{resolution}', data=features)
    root.create_dataset(f'ScanMask/{resolution}', data=scan_mask)
    
    print(f"Dummy Zarr file created at {path} with {num_samples} samples and {input_dim} features.")


def generate_dummy_targets(num_samples, num_classes=2):
    """
    Generate random targets for a classification task.

    Parameters:
    ----------
    num_samples : int
        The number of samples (length of targets).
    num_classes : int
        The number of possible classes (e.g., 2 for binary, 3+ for multiclass).

    Returns:
    -------
    targets : torch.Tensor
        A tensor of shape (num_samples,) containing random class labels.
    """
    # Generate random class labels (integers) between 0 and num_classes - 1
    targets = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Convert to torch Tensor
    return torch.tensor(targets, dtype=torch.long)


@pytest.mark.skip()
def test_create_zarr():

    # Create dummy zarr data
    zarr_path = 'dummy_data.zarr'
    create_dummy_zarr(zarr_path)
    assert os.path.exists(zarr_path)

    clean_up(shutil.rmtree(zarr_path))


@pytest.mark.skip()
def test_run_training():

    # Create dummy zarr data
    zarr_path = 'dummy_data.zarr'
    create_dummy_zarr(zarr_path, num_samples=200, input_dim=5)

    # Create targets
    multiclass_targets = generate_dummy_targets(num_samples=200, num_classes=2)

    # Run training
    run_training(
        zarr_path='dummy_data.zarr',  # Dummy Zarr file path
        resolution='1',
        hidden_dims=[128],
        num_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        targets=multiclass_targets  # Pass the generated targets
    )

    # Cleanup
    clean_up(zarr_path)


def test_multiple_checkpoints():
    """
    Test that checkpoints are saved in both primary and backup directories.
    """
    # Set up paths for primary and backup directories
    primary_dir = "test_models"
    backup_dir = "backup_models"

    # Create dummy zarr data
    zarr_path = 'dummy_data.zarr'
    create_dummy_zarr(zarr_path, num_samples=200, input_dim=5)

    # Create targets
    multiclass_targets = generate_dummy_targets(num_samples=200, num_classes=2)

    # Run training
    run_training(
        zarr_path='dummy_data.zarr', 
        resolution='1',
        hidden_dims=[128],
        num_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        targets=multiclass_targets,
        checkpointing={
            "dirname": primary_dir,
            "backup_location": backup_dir,
            "filename_prefix": "best",
            "n_saved": 2,
            "score_function": lambda engine: -engine.state.metrics['loss'],
            "score_name": "val_loss"
        }
    )

    # Check that checkpoints exist in both directories
    primary_files = sorted(os.listdir(primary_dir))
    backup_files = sorted(os.listdir(backup_dir))

    assert len(primary_files) > 0, "No checkpoint files found in primary directory."
    assert primary_files == backup_files, "Mismatch between primary and backup checkpoint files."

    print("Test passed: Checkpoints saved in both primary and backup directories.")

    # Cleanup after test
    clean_up([primary_dir, backup_dir, zarr_path])