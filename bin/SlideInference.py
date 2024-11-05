import zarr
import pandas as pd
import torch
import argparse
from os import path
import numpy as np


def slide_inference(image_ids_file, zarr_dir_path, model, agg_type, scan_ds, metadata_path, results_file_name, results_path, class_order=None):
    test_ids=pd.read_csv(image_ids_file)
    metadata_df=pd.read_csv(metadata_path)
    test_metadata_df=metadata_df.loc[metadata_df["SlideID"].isin(test_ids["SlideID"]),:]
    test_ids=test_ids.to_numpy()
    image_id=test_ids[0,0]
    print(path.join(zarr_dir_path,f"ViT_features_{image_id}.zarr"))
    z=zarr.open(path.join(zarr_dir_path,f"ViT_features_{image_id}.zarr"),"r")
    scan_features=z[f"SlideLevelFeatures/{agg_type}/{scan_ds}"][:]
    scan_features=torch.tensor(scan_features).reshape(1,-1).float()
    for image_id in test_ids[1:,0]:
        print(path.join(zarr_dir_path,f"ViT_features_{image_id}.zarr"))
        z=zarr.open(path.join(zarr_dir_path,f"ViT_features_{image_id}.zarr"),"r")
        scan_features_sub=z[f"SlideLevelFeatures/{agg_type}/{scan_ds}"][:]
        scan_features_sub=torch.tensor(scan_features_sub).reshape(1,-1).float()
        scan_features=torch.concat((scan_features,scan_features_sub),dim=0)
    with torch.inference_mode():
        m_out=model(scan_features)
    out_channels=m_out.detach().numpy()
    if out_channels.ndim < 2:
        out_channels=out_channels.reshape(-1,1)
    out_channels_df=pd.DataFrame(out_channels)
    if class_order is not None:
        channel_names=[f"{c}_prob" for c in class_order]
    else:
        channel_names=["y_hat"]
    metadata_names=test_metadata_df.columns.to_list()
    test_metadata_df=pd.concat((test_metadata_df.reset_index(drop=True),out_channels_df.reset_index(drop=True)),axis=1)
    print(metadata_names+channel_names)
    test_metadata_df.columns=metadata_names+channel_names
    test_metadata_df.to_csv(f"{results_file_name}.csv",index=False)
    test_metadata_df.to_csv(path.join(results_path,f"{results_file_name}.csv"),index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("image_ids_file",type=str)
    parser.add_argument("zarr_dir_path",type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("agg_type", type=str)
    parser.add_argument("scan_ds",type=int)
    parser.add_argument("class_order", type=str)
    parser.add_argument("metadata_path", type=str)
    parser.add_argument("results_file_name", type=str)
    parser.add_argument("results_path", type=str)
    args=parser.parse_args()

    #ToDo:replace with loading head from trainer
    m = torch.jit.load(args.model_name)
    m.eval()
    class_order = args.class_order.split(',') if args.class_order else None

    slide_inference(args.image_ids_file, args.zarr_dir_path, m, args.agg_type, args.scan_ds, args.metadata_path, args.results_file_name, args.results_path, class_order)
    