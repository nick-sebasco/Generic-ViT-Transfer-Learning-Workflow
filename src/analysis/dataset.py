import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from typing import Tuple, Optional
import pandas as pd
import os


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

    def __init__(self, image_ids, feature_zarr_dir, agg_type, scan_ds, meta_data_path: Optional[str] = None, target_column: Optional[str]=None, class_order: Optional[torch.Tensor]=None, ordinal: Optional[bool]=False):
        self.image_ids = image_ids
        self.feature_zarr_dir=feature_zarr_dir
        self.zarr_idx=f"SlideLevelFeatures/{agg_type}/{scan_ds}"
        if meta_data_path is not None:
            if target_column is None:
                raise ValueError("If metadata_path is provided target_column must be as well")
            meta_df=pd.read_csv(meta_data_path)
            id_idx=[]
            for id in image_ids:
                id_idx.append(meta_df.index[meta_df["SlideID"]==id].to_list()[0])
            meta_targets=meta_df.loc[id_idx,target_column].to_numpy()
            if class_order:
                targets=torch.zeros((len(image_ids),len(class_order)))
                for ii,id in enumerate(image_ids):
                    targets[ii,:]=np.equal(class_order,id)
                    if ordinal and np.any(targets[ii:]):
                        targets[ii,:np.where(targets[ii,:])[0][0]]=1
            else:
                targets=meta_targets.reshape((-1,1))
        self.targets = torch.tensor(targets).view

    def __len__(self) -> int:
        return self.features.shape[1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image_id=self.image_ids[idx]
        z=zarr.open(os.path.join(self.feature_zarr_dir,f"ViT_features_{image_id}.zarr"),"r")
        x=z[self.zarr_idx][:,:]
        x=torch.tensor(x)
        t=self.targets[idx,:]
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


def load_targets(meta_data_path: str, image_id: str):
    """
    """