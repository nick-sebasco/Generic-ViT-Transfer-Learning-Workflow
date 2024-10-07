import torch
from torch.utils.data import Dataset
import zarr


class ZarrDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[1]

    def __getitem__(self, idx):
        x = self.features[:, idx]
        t = self.targets[idx] if self.targets is not None else None
        return x, t

def load_features(zarr_path, resolution):
    features_zarr = zarr.open(zarr_path, mode='r')
    features = features_zarr[f'Features/{resolution}']
    scan_mask = features_zarr[f'ScanMask/{resolution}']
    valid_positions = scan_mask[:] == 1
    valid_features = features[:, valid_positions]
    return valid_features