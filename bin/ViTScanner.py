import zarr
import torch
import torchvision
import pandas as pd
from jsonargparse import CLI
import numpy as np
import warnings

#To-Do figure out how to set the arcetecture via string
def vitScan(image_id : str, scan_group_csv : str, image_dir : str, feature_dir : str, scan_step : int, scan_ds : int, patch_size : int, model_checkpoint : str = None, device : str = 'auto', scan_mask : bool = False):
    """
    Scans selected patches of an image using a pretrained pytorch model and saves features to a prepaired zarr file. 

    Parameters
    -------------
    image_id : id of target slide
    scan_group_csv : path to csv containing patch coordinates for scanning
    image_dir : path to directory of zarrs converted from slides
    feature_dir : path to directory of zarrs storing the ViT feature representstion
    scan_step : size of step between patches
    scan_ds : number of times to downsample before scanning (scan patch will be 1/2^(scan_ds) the resolution of the initial image)
    patch_size : size of input to the model
    model_checpoint : path to .pth file with alternate weights
    device : target torch device, must be one of ['auto', 'cuda', 'cpu']
    scan_mask : create a mask of which patches have been scanned (useful for sanity checking the scan but creates an ineficent zarr)
    """
    if device=="auto":
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"
    elif device=="cuda":
        if not torch.cuda.is_available():
            warnings.warn("GPU unavailable, using CPU")
            device="cpu"
    elif device!="cpu":
        raise ValueError("device must be one of ['auto', 'cuda', 'cpu']")

    model=torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,progress=False)

    model.heads=torch.nn.Sequential(torch.nn.Identity())
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    image_path=f"{image_dir}slide_{image_id}.zarr"
    feature_path=f"{feature_dir}ViT_features_{image_id}.zarr"
    zi=zarr.open(image_path,mode='r+')
    image=zi[f"0/{scan_ds}"]
    zf=zarr.open(feature_path)
    scan_groups_df=pd.read_csv(scan_group_csv)
    scan_groups=scan_groups_df.groupby("Batch")
    for _,scan_group in scan_groups:
        batch=np.zeros(len(scan_group,3,patch_size,patch_size))
        for ii, row in enumerate(scan_group.itertuples()):
            batch[ii,:,:,:]=np.copy(image[0,:,0,row.IPos:(row.IPos+patch_size),row.JPos:(row.JPos+patch_size)])
        batch=torch.Tensor(batch,device=device)
        with torch.inference_mode():
            features=model(batch)
        features=features.detach()
        for ii, row in enumerate(scan_group.itertuples()):
            ipos=row.IPos/scan_step
            jpos=row.JPos/scan_step
            zf[f"Features/{scan_ds}"][:,ipos,jpos]=features[ii,:].numpy()
            if scan_mask:
                zf[f"ScanMask/{scan_ds}"][ipos,jpos]=1
    return None


    

if __name__=="__main__":
    CLI(vitScan)