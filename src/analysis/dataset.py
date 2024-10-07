import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from typing import Tuple, Optional


class ZarrDataset(Dataset):
    """
    A PyTorch Dataset class that loads features and corresponding targets from Zarr arrays.

    Parameters
    ----------
    features : np.ndarray or torch.Tensor
        The feature data, typically a 2D array where the first dimension is the number of features
        and the second dimension is the number of samples.
    targets : Optional[np.ndarray or torch.Tensor]
        The corresponding target data, typically a 1D array where each element corresponds to
        a target value for the associated sample. If targets are not provided, the dataset
        will only return the features.
    """

    def __init__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None):
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.features[:, idx]
        t = self.targets[idx] if self.targets is not None else None
        return x, t


def load_features(zarr_path: str, resolution: str) -> np.ndarray:
    """
    Loads feature data from a Zarr array and filters valid positions using a scan mask.

    Parameters
    ----------
    zarr_path : str
        The file path to the Zarr dataset.
    resolution : str
        The resolution level of the features to load from the Zarr dataset.

    Returns
    -------
    np.ndarray
        A 2D array of valid features where invalid positions have been filtered out based on the scan mask.
    """
    # Open the Zarr store
    features_zarr = zarr.open(zarr_path, mode='r')
    
    # Load features and scan mask
    features = features_zarr[f'Features/{resolution}']
    scan_mask = features_zarr[f'ScanMask/{resolution}']
    
    # Convert to NumPy arrays
    features_np = np.array(features)
    scan_mask_np = np.array(scan_mask)
    
    # Select only valid positions using the scan mask
    valid_features = features_np[:, scan_mask_np]
    
    return valid_features
