import zarr
from skimage.filters import gaussian
from skimage import measure
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import argparse


class GuasianTissueThresholder(nn.Module):
    def __init__(self,thresh,channels,agg_function,sigma=10):
        self.thresh=thresh
        self.channels=channels
        self.agg_function=agg_function
        self.sigma=sigma
    def forward(self,patch):
        patch_sub=np.copy(patch[0,self.channels,0,:,:])
        patch_sub=self.agg_function(patch_sub)
        patch_sub=gaussian(patch_sub,sigma=self.sigma)
        return np.less_equal(patch_sub,self.thresh) 
        



def scanBatching(image_id,image_dir,roi_mask_dir,feature_dir,roi_type,roi_area_bounds, vit_model_dim, scan_step, batch_size, scan_ds, roi_mask_ds, roi_identifier_model, complete_scan=False, compute_rois=False):
    image_path=f"{image_dir}slide_{image_id}.zarr"
    roi_mask_path=f"{roi_mask_dir}roi_masks_{image_id}.zarr"
    feature_path=f"{feature_dir}ViT_features_{image_id}.zarr"
    zi=zarr.open(image_path)["0"]
    zf=zarr.open(feature_path, mode='a')
    zf.zeros(f"{scan_ds}",(vit_model_dim,zi[f"{scan_ds}"].shape[3],zi[f"{scan_ds}"].shape[4]))
    if  complete_scan:
        class Section:
            bbox=[0,0,zi[f"{scan_ds}"].shape[3],zi[f"{scan_ds}"].shape[4]]
        sections=[Section()]
        skip_sections=[0]
    else:
        if not(os.path.exists(roi_mask_path)) or compute_rois:
            image_ds=zarr.open(image_path)[f'0/{roi_mask_ds}']
            roi_mask=roi_identifier_model(image_ds)
            zm=zarr.open(roi_mask_path, mode='a')
            zm.array(f"{roi_type}/{roi_mask_ds}",roi_mask)
            
        roi_mask=zarr.open(roi_mask_path)[roi_type]

        if np.max(roi_mask)>1:
            section_mask=roi_mask
        else:
            section_mask=measure.label(roi_mask)
        sections=measure.regionprops(section_mask)
        skip_sections=np.ones(len(sections),dtype=bool)
        for ii, sec in enumerate(sections):
            if sec.area>roi_area_bounds[0] and sec.area<roi_area_bounds[1]:
                skip_sections[ii]=0
    patch_i=[]
    patch_j=[]
    patch_count=0
    batch_itt=0
    for ii, sec in enumerate(sections):
        if skip_sections[ii]:
            continue
        bbox=sec.bbox*2**(roi_mask_ds-scan_ds)
        cur_i=bbox[0]-bbox[0]%scan_step
        cur_j=bbox[1]-bbox[0]%scan_step
        
        
        while cur_i<bbox[2]:
            while cur_j<bbox[3]:
                if patch_count==batch_size:
                    batch_df=pd.DataFrame.from_dict({"IPos":patch_i,"JPos":patch_j})
                    batch_df.to_csv(f"{image_id}_batch_{batch_itt}.csv",index=False)
                    patch_count=0
                    batch_itt+=1
                    patch_i=[]
                    patch_j=[]

                patch_i=patch_i.append(cur_i)
                patch_j=patch_j.append(cur_j)
                cur_i=cur_i+scan_step
                cur_j=cur_j+scan_step
                patch_count+=1
    return 1

if __name__=="__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument("image_id",type=str)
    CLI.add_argument("image_dir",type=str)
    CLI.add_argument("feature_dir",type=str)
    CLI.add_argument("roi_mask_dir",type=str)
    CLI.add_argument("roi_type",type=str)
    CLI.add_argument("roi_area_bounds_min",type=int)
    CLI.add_argument("roi_area_bounds_max",type=int)
    CLI.add_argument("vit_model_dim",type=int)
    CLI.add_argument("scan_step",type=int)
    CLI.add_argument("batch_size",type=int)
    CLI.add_argument("scan_ds",type=int) 
    CLI.add_argument("roi_mask_ds",type=int)
    CLI.add_argument("complete_scan",type=bool) 
    CLI.add_argument("compute_rois",bool)
    CLI.add_argument("roi_identifier_model",type=str,choices=["ChannelMeanThresh","GreenThresh"])
    CLI.add_argument("roi_thresh",type=float)

args=CLI.parse_args()
if args.roi_identifier_model=="ChannelMeanThresh":
    id_model=GuasianTissueThresholder(args.roi_thresh,[0,1,2],lambda x:np.mean(x,axis=0),10)
if args.roi_identifier_model=="ChannelMeanThresh":
    id_model=GuasianTissueThresholder(args.roi_thresh,[1],lambda x:x,10)
scanBatching(args.image_id,args.image_dir,args.feature_dir, args.roi_mask_dir,args.roi_type,[args.roi_area_bounds_min,args.roi_area_bounds_max], args.vit_model_dim, args.scan_step, args.batch_size, args.scan_ds, args.roi_mask_ds, id_model, args.complete_scan, args.compute_rois)    


    



    

