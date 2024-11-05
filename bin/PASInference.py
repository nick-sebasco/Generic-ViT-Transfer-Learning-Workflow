import zarr
import pandas as pd
import torch
import argparse
from os import path
import numpy as np


def inference(image_id, zarr_dir_path, model, scan_ds, num_out_channels):
    z=zarr.open(path.join(zarr_dir_path,f"ViT_features_{image_id}.zarr"))
    scan_mask=z[f"ScanMask/{scan_ds}"][:,:]
    scan_features=z[f"Features/{scan_ds}"]
    idxs=np.where(scan_mask)
    out_channels=np.zeros((len(idxs[0]),num_out_channels))
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
    patch_out_df=pd.DataFrame(out_channels,columns=[f"X{ch}" for ch in range(num_out_channels)])
    patch_meta_df=pd.DataFrame.from_dict({"SlideID":image_id_list,"YPos":i_pos,"XPos":j_pos})
    patch_df=pd.concat((patch_meta_df,patch_out_df),axis=1)
    patch_df.to_csv(f"Patch_Scores_{image_id}.csv",index=False)
    mean_out_df=pd.DataFrame(mean_out,columns=[f"X{ch}" for ch in range(num_out_channels)])
    mean_meta_df=pd.DataFrame.from_dict({"SlideID":[image_id]})
    mean_df=pd.concat((mean_meta_df,mean_out_df),axis=1)
    mean_df.to_csv(f"Slide_Scores_{image_id}.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("image_id",type=str)
    parser.add_argument("zarr_dir_path",type=str)
    parser.add_argument("scan_ds",type=int)
    args=parser.parse_args()

    #ToDo:replace with loading head from trainer
    state_dict=torch.load("/projects/korstanje-lab/MachineLearningModelWeights/PASKidneyViTAgeClassifierWeights.pth",map_location=torch.device('cpu'))
    m=torch.nn.Sequential(torch.nn.Linear(1280,2),torch.nn.Sigmoid())
    m_state_dict={'0.weight':state_dict['model']["module.heads.head.weight"],'0.bias':state_dict['model']["module.heads.head.bias"]}
    m.load_state_dict(m_state_dict)

    inference(args.image_id, args.zarr_dir_path, m, args.scan_ds, 2)
    