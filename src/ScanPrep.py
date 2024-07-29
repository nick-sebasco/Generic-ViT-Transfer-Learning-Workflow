import zarr
from skimage.filters import gaussian
from skimage import measure
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse


class GuasianTissueThresholder(nn.Module):
    """
    A module for identifying tissue using a guasian blur+threshold, 
    handelded as a module so alternate options can be added in the future if desired.

    Atributes
    ---------------
    thresh : float
        value between 0 & 1 indicating the intensity treshold ove whice a pixel is considered white background
    channels : tuple of ints
        indicates which channels to include in the intensity calculation
    agg_function : funtion
        Takes a 5D array in [T,C,Z,Y,X] order (default ome ordering) as input and returns a 2D intensity array
    sigma : float
        sigma value for the gausian blur
    
    Methods
    ----------
    forward(arr)
        Applies a gausian blur and treshold to detect tissue in a bright field image
    """
    def __init__(self,thresh,channels,agg_function,sigma=10):
        """
        Constructs module

        Parameters
        ---------------
        thresh : float
            value between 0 & 1 indicating the intensity treshold ove whice a pixel is considered white background
        channels : tuple of ints
            indicates which channels to include in the intensity calculation
        agg_function : funtion
            Takes a 5D array in [T,C,Z,Y,X] order (default ome ordering) as input and returns a 2D intensity array
        sigma : float
            sigma value for the gausian blur
        """
        self.thresh=thresh
        self.channels=channels
        self.agg_function=agg_function
        self.sigma=sigma
    def forward(self,arr):
        """
        Applies a gausian blur and treshold to detect tissue in a bright field image

        Parameters
        ---------------
        arr : array
            5D array in default OME order (T,C,Z,Y,X)
        
        Returns
        ---------------
        Intensity map of arr[0,:,0,:,:] based on the module attributes  
        """
        arr_sub=np.copy(arr[0,self.channels,0,:,:])
        arr_sub=self.agg_function(arr_sub)
        arr_sub=gaussian(arr_sub,sigma=self.sigma)
        return np.less_equal(arr_sub,self.thresh) 
        



def findRegions(roi_mask, roi_min_area=0, roi_max_area=np.inf):
    """
    Finds connecected component regions in provided roi mask and filters based on area

    Parameters
    -------------
    roi_mask: array of either booleans or ints
        array indicating whether a givin pixel is part of a target object, or a labeled array of target objects
    roi_min_area: int
        minimum required area in number of pixels for an object to be scanned
    roi_max_area: int
        maximum permitted area in number of pixels for an object to be scanned    

    Returns
    ------------
    dataframe of target regions for the scanning model
    """
    
    if np.max(roi_mask)>1:
        section_mask=roi_mask
    else:
        section_mask=measure.label(roi_mask)
    sections_dict=measure.regionprops_table(section_mask,separator='-')
    sections_df=pd.DataFrame.from_dict(sections_dict)
    keep_sections=np.logical_and(np.greater_equal(sections_df["area"],roi_min_area),np.less_equal(sections_df["area"],roi_max_area))
    return sections_df.loc[keep_sections,:]

def scanBatching(image_id, sections, scan_step, batch_size, scan_group_size, scan_ds, roi_mask_ds):
    """
    Breaks the Bounding Box for each ROI into patches to be scanned and assigns each patch to a batch and scan group for scanning 
    and saves as a csv for each scan group

    Parameters
    -------------
    image_id : str
        id of target slide
    sections : DataFrame from skimage.measure.regionprops_tabel
        Section DataFrame containing bounding boxes of the regions to be scanned
    scan_step : int
        size of step between scan patches
    batch_size : int
        number of patches per batch
    scan_group_size : int
        number of batches to be handeled by each scan process 
    scan_ds : int
        number of times to downsample befor scanning (scan patch will be 1/2^(scan_ds) the resolution of the initial image)
    roi_mask_ds
        number of times downssampled before identifying the regions to be scanned
    
    """
    
    patch_i=[]
    patch_j=[]
    batch=[]
    patch_count=0
    batch_itt=0
    scan_group_itt=0
    for ii, sec in sections.iterrows():
        bbox=np.array([sec.bbox-0,sec.bbox-1,sec.bbox-2,sec.bbox-3])
        bbox=bbox*2**(roi_mask_ds-scan_ds)
        cur_i=bbox[0]-bbox[0]%scan_step
        cur_j=bbox[1]-bbox[0]%scan_step
        
        
        while cur_i<bbox[2]:
            while cur_j<bbox[3]:
                if patch_count==batch_size*scan_group_size:
                    batch_df=pd.DataFrame.from_dict({"IPos":patch_i,"JPos":patch_j,"Batch":batch})
                    batch_df.to_csv(f"{image_id}_scan_group_{scan_group_itt}.csv",index=False)
                    patch_count=0
                    batch_itt=0
                    scan_group_itt+=1
                    patch_i=[]
                    patch_j=[]
                    batch=[]

                patch_i=patch_i.append(cur_i)
                patch_j=patch_j.append(cur_j)
                batch=batch.append(batch_itt)
                cur_i=cur_i+scan_step
                cur_j=cur_j+scan_step
                patch_count+=1
                if patch_count==batch_size:
                    batch_itt+=1
    batch_df=pd.DataFrame.from_dict({"IPos":patch_i,"JPos":patch_j,"Batch":batch})
    batch_df.to_csv(f"{image_id}_scan_group_{scan_group_itt}.csv",index=False)
    return None



if __name__=="__main__":
    CLI=argparse.ArgumentParser(
                        prog="ViT Scan Prep",
                        description="""prepairs for scanning an image with the ViTScanner
                          by identifying regions to scan (if nessesary)
                          and pre batching the image patches""")
    CLI.add_argument("image_id",type=str, help="id of target slide")
    CLI.add_argument("image_dir",type=str, help="path to directory of zarrs converted from slides")
    CLI.add_argument("feature_dir",type=str, help="path to directory of zarrs storing the ViT feature representstion")
    CLI.add_argument("roi_mask_dir",type=str, help="path to directory of zarrs storing ROI masks of the slides")
    CLI.add_argument("roi_type",type=str, help="label indicating which ROI mask should be used to determine the regions to scan")
    CLI.add_argument("roi_min_area",type=int, help="minimum required area in number of pixels for an object to be scanned")
    CLI.add_argument("roi_max_area",type=int, help="maximum permitted area in number of pixels for an object to be scanned")
    CLI.add_argument("vit_model_channels",type=int, help="numper of channels output by the headless ViT")
    CLI.add_argument("scan_step",type=int, help="size of step between scan patches")
    CLI.add_argument("batch_size",type=int, help="number of patches per batch")
    CLI.add_argument("scan_group_size",type=int, help="number of batches to be handeled by each scan process ")
    CLI.add_argument("scan_ds",type=int, help="number of times to downsample befor scanning (scan patch will be 1/2^(scan_ds) the resolution of the initial image)") 
    CLI.add_argument("roi_mask_ds",type=int, help="number of times downssampled before identifying the regions to be scanned")
    CLI.add_argument("complete_scan",type=bool, help="flag indicating if the whole image should be scanned ignoring ROIs") 
    CLI.add_argument("compute_rois",type=bool, help="flag indicating if existing ROIs should be recomputed")
    CLI.add_argument("roi_identifier_model",type=str,choices=["ChannelMeanThresh","GreenThresh"], help="what model to use for identifying ROIs if not precomputed and compute_rois==False")
    CLI.add_argument("roi_thresh",type=float, help="Threshold for simple ROI detection models")

    args=CLI.parse_args()

    image_path=f"{args.image_dir}slide_{args.image_id}.zarr"
    roi_mask_path=f"{args.roi_mask_dir}roi_masks_{args.image_id}.zarr"
    feature_path=f"{args.feature_dir}ViT_features_{args.image_id}.zarr"

    #pre-create zarr array to contain feature scan (prevents race condition issiues that would arise if initialized at scan time)
    zi=zarr.open(image_path)["0"]
    zf=zarr.open(feature_path, mode='a')
    zf.zeros(f"{args.scan_ds}",(args.vit_model_channels,int(np.ceil(zi[f"{args.scan_ds}"].shape[3]/args.scan_step)),int(np.ceil(zi[f"{args.scan_ds}"].shape[4]/args.scan_step))),chunks=(args.vit_model_channels,1,1))

    # compute roi mask if nessasary or indicated by parameters
    zm=zarr.open(roi_mask_path)
    if args.compute_rois or (f"{args.roi_type}/{args.roi_mask_ds}" not in zm):
        if args.roi_identifier_model=="ChannelMeanThresh":
            id_model=GuasianTissueThresholder(args.roi_thresh,[0,1,2],lambda x:np.mean(x,axis=0),10)
        if args.roi_identifier_model=="ChannelMeanThresh":
            id_model=GuasianTissueThresholder(args.roi_thresh,[1],lambda x:x,10)
        image_ds=zarr.open(image_path)[f'0/{args.roi_mask_ds}']
        roi_mask=id_model(image_ds)
        zm.array(f"{args.roi_type}/{args.roi_mask_ds}",roi_mask)   
    else:
        roi_mask=zm[f"{args.roi_type}/{args.roi_mask_ds}"][:,:]
    
    # find roi regions
    sections=findRegions(roi_mask,args.roi_min_area,args.roi_max_area)

    #create scan group csvs
    scanBatching(args.image_id, sections, args.scan_step, args.batch_size, args.scan_group_size, args.scan_ds, args.roi_mask_ds)






    



    

