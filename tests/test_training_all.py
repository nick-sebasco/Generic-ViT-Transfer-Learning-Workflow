import zarr
import numpy as np
import torch 
import pytest
import os
import shutil

from src.analysis.training import run_training


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


    try:
        shutil.rmtree(zarr_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


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
        num_epochs=5,  # Quick test with 10 epochs
        batch_size=32,
        learning_rate=0.001,
        targets=multiclass_targets  # Pass the generated targets
    )


    # Cleanup
    try:
        shutil.rmtree(zarr_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))