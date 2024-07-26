import zarr
import torch
import dask
import torchvision

def vitScan(image_path,feature_path, daskparameters..., ModelClass=torchvision.models.vit_h_14 checkpoint=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1):
    model=ModelClass(weights=checkpoint,progress=False)
    
    z = da.from_zarr(image_path,)
