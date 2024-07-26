import zarr
import torch
import torchvision

def vitScan(image_id,image_dir,feature_path, batch_csv, scan_ds, ModelClass=torchvision.models.vit_h_14, checkpoint=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1):
    model=ModelClass(weights=checkpoint,progress=False)
    zi=zarr.open(image)


