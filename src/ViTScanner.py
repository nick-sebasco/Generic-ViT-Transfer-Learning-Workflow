import zarr
import torch
import torchvision
import pandas as pd
import argparse
import numpy as np


def vitScan(image_id,image_dir,feature_dir, scan_group_csv, scan_step, scan_ds, patch_size, model, device):
    """
    Scans selected patches of an image using a pretrained pytorch model and saves features to a prepaired zarr file. 

    Parameters
    -------------
    image_id : str
        id of target slide
    image_dir : str
        path to directory of zarrs converted from slides
    feature_dir : str
        path to directory of zarrs storing the ViT feature representstion
    scan_group_csv : str
        path to csv containing patch coordinates for scanning
    scan_step : str
        size of step between patches
    scan_ds : int
        number of times to downsample before scanning (scan patch will be 1/2^(scan_ds) the resolution of the initial image)
    patch_size : str
        size of input to the model
    model : pytorch model 
        model for scanning
    device : str
        target torch device
    """
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
            zf[f"{scan_ds}"][:,ipos,jpos]=features[ii,:].numpy()
    return None


    

if __name__=="__main__":
    CLI=argparse.ArgumentParser(
                        prog="ViT Scanner",
                        description="""scans target patches of an image with indicated model""")
    CLI.add_argument("image_id",type=str, help="id of target slide")
    CLI.add_argument("image_dir",type=str, help="path to directory of zarrs converted from slides")
    CLI.add_argument("feature_dir",type=str, help="path to directory of zarrs storing the ViT feature representstion")
    CLI.add_argument("scan_group_csv",type=str, help="path to csv containing patch coordinates for scanning")
    CLI.add_argument("scan_step",type=int, help="size of step between patches")
    CLI.add_argument("scan_ds",type=int, help="number of times to downsample before scanning (scan patch will be 1/2^(scan_ds) the resolution of the initial image)")
    CLI.add_argument("patch_size",type=str,help="size of input to the model")
    CLI.add_argument("device",type=str, choices=["auto","cpu","cuda"],help="target torch device")
    CLI.add_argument("-p","--model_checkpoint",type=str, help="path to a pth file with alternate pretrained weights")
    CLI.add_argument("-a","--model_archetecture",type=str, help="not implemented:have to figure out how to implement with CLI input")
    args=CLI.parse_args()

    if args.device=="auto":
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"
    else:
        device=args.device

    model=torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,progress=False)

    model.heads=torch.nn.Sequential(torch.nn.Identity())
    if args.model_checkpoint:
        model.load_state_dict(torch.load(args.model_checkpoint))
    model.to(device)
    vitScan(args.image_id,args.image_dir, args.feature_dir, args.scan_group_csv, args.scan_step, args.scan_ds, args.patch_size, model, device)
