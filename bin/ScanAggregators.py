import zarr
import numpy as np
import argparse
from os import path

def mean_agg(zarr_path,scan_ds,agg_window=500):
    z=zarr.open(zarr_path)
    scan_mask=scan_mask=z[f"ScanMask/{scan_ds}"]
    scan_features=z[f"Features/{scan_ds}"]
    i_start=0
    j_start=0
    sum_features=np.zeros(scan_features.shape[0])
    num_patches=0
    while i_start<scan_mask.shape[0]:
        while j_start<scan_mask.shape[1]:
            if i_start+agg_window>scan_mask.shape[0]:
                i_stop=scan_mask.shape[0]
            else:
                i_stop=i_start+agg_window
            if j_start+agg_window>scan_mask.shape[1]:
                j_stop=scan_mask.shape[1]
            else:
                j_stop=j_start+agg_window
            scan_mask_sub=scan_mask[i_start:i_stop,j_start:j_stop]
            scan_features_sub=scan_features[:,i_start:i_stop,j_start:j_stop]
            num_patches+=np.sum(scan_mask_sub)
            patch_idxs=np.where(scan_mask_sub)
            sum_features+=np.sum(scan_features_sub[:,patch_idxs[0]+i_start,patch_idxs[1]+j_start])
            j_start+=agg_window
        j_start=0
        i_start+=agg_window
    mean_features=sum_features/num_patches
    z[f"SlideLevelFeatures/Mean/{scan_ds}"]=mean_features

def mean_sdev_agg(zarr_path,scan_ds,agg_window=500):
    z=zarr.open(zarr_path)
    scan_mask=scan_mask=z[f"ScanMask/{scan_ds}"]
    scan_features=z[f"Features/{scan_ds}"]
    i_start=0
    j_start=0
    sum_features=np.zeros(scan_features.shape[0])
    num_patches=0
    while i_start<scan_mask.shape[0]:
        while j_start<scan_mask.shape[1]:
            if i_start+agg_window>scan_mask.shape[0]:
                i_stop=scan_mask.shape[0]
            else:
                i_stop=i_start+agg_window
            if j_start+agg_window>scan_mask.shape[1]:
                j_stop=scan_mask.shape[1]
            else:
                j_stop=j_start+agg_window
            scan_mask_sub=scan_mask[i_start:i_stop,j_start:j_stop]
            scan_features_sub=scan_features[:,i_start:i_stop,j_start:j_stop]
            num_patches+=np.sum(scan_mask_sub)
            patch_idxs=np.where(scan_mask_sub)
            sum_features+=np.sum(scan_features_sub[:,patch_idxs[0]+i_start,patch_idxs[1]+j_start],axis=1)
            j_start+=agg_window
        j_start=0
        i_start+=agg_window
    mean_features=sum_features/num_patches
    sum_deviation=np.zeros(scan_features.shape[0])
    while i_start<scan_mask.shape[0]:
        while j_start<scan_mask.shape[1]:
            if i_start+agg_window>scan_mask.shape[0]:
                i_stop=scan_mask.shape[0]
            else:
                i_stop=i_start+agg_window
            if j_start+agg_window>scan_mask.shape[1]:
                j_stop=scan_mask.shape[1]
            else:
                j_stop=j_start+agg_window
            scan_mask_sub=scan_mask[i_start:i_stop,j_start:j_stop]
            scan_features_sub=scan_features[:,i_start:i_stop,j_start:j_stop]
            patch_idxs=np.where(scan_mask_sub)
            deviations=np.square(scan_features_sub[:,patch_idxs[0]+i_start,patch_idxs[1]+j_start]-mean_features.reshape((-1,1)))
            sum_deviation+=np.sum(deviations,axis=1)
            j_start+=agg_window
        j_start=0
        i_start+=agg_window
    sdev_features=np.sqrt(sum_deviation/(num_patches-1))
    mean_sdev_features=np.concatenate((mean_features,sdev_features))
    z[f"SlideLevelFeatures/Mean_SDev/{scan_ds}"]=mean_sdev_features

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("image_id",type=str)
    parser.add_argument("feature_dir_path",type=str)
    parser.add_argument("scan_ds",type=int)
    parser.add_argument("agg_type",type=str,choices=("mean_agg","mean_sdev_agg"))
    parser.add_argument("agg_window",type=int)
    args=parser.parse_args()
    
    zarr_path=path.join(args.feature_dir_path,f"ViT_features_{args.image_id}.zarr")
    globals()[args.agg_type](zarr_path,args.scan_ds,args.agg_window)



