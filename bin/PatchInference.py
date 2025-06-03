import zarr
import pandas as pd
import torch
import argparse
from os import path
import numpy as np


def patch_inference(image_id, zarr_dir_path, model, scan_ds, class_order=None):
    z=zarr.open(path.join(zarr_dir_path,f"ViT_features_{image_id}.zarr"))
    scan_mask=z[f"ScanMask/{scan_ds}"][:,:]
    scan_features=z[f"Features/{scan_ds}"]
    idxs=np.where(scan_mask)
    if class_order is not None:
        out_channels=np.zeros((len(idxs[0]),len(class_order)))
    else:
        out_channels=np.zeros((len(idxs[0]),1))
    i_pos=np.zeros(len(idxs[0]),dtype=np.uint32)
    j_pos=np.zeros(len(idxs[0]),dtype=np.uint32)
    image_id_list=[image_id]*len(idxs[0])
    for nn,(ii,jj) in enumerate(zip(idxs[0],idxs[1])):
        i_pos[nn]=ii
        j_pos[nn]=jj
        f=scan_features[:,ii,jj]
        f.reshape((1,-1))
        f=torch.Tensor(f)
        f_out=model(f)
        out_channels[nn,:]=np.squeeze(f_out.detach().numpy())
    mean_out=np.mean(out_channels,axis=0,keepdims=True)
    if class_order is not None:
        column_names=class_order
    else:
        column_names=["y_hat"]
    patch_out_df=pd.DataFrame(out_channels,columns=column_names)
    patch_meta_df=pd.DataFrame.from_dict({"SlideID":image_id_list,"YPos":i_pos,"XPos":j_pos})
    patch_df=pd.concat((patch_meta_df,patch_out_df),axis=1)
    patch_df.to_csv(f"Patch_Scores_{image_id}.csv",index=False)
    mean_out_df=pd.DataFrame(mean_out,columns=column_names)
    mean_meta_df=pd.DataFrame.from_dict({"SlideID":[image_id]})
    mean_df=pd.concat((mean_meta_df,mean_out_df),axis=1)
    mean_df.to_csv(f"Mean_Patch_Scores_{image_id}.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("image_id",type=str)
    parser.add_argument("zarr_dir_path",type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("scan_ds",type=int)
    parser.add_argument("class_order", type=str)
    args=parser.parse_args()

    
    m = torch.jit.load(args.model_name)
    m.eval()

    class_order = args.class_order.split(',') if args.class_order else None
    patch_inference(args.image_id, args.zarr_dir_path, m, args.scan_ds, class_order)
    