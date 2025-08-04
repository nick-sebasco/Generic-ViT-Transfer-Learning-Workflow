import zarr
from skimage.filters import gaussian
from skimage import measure
import torch.nn as nn
import numpy as np
import pandas as pd
from jsonargparse import CLI



class GuasianTissueThresholder:
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
    def __call__(self,arr):
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
        if np.max(arr_sub)>1:
            arr_sub=arr_sub/255
        return np.less_equal(arr_sub,self.thresh)
        



def findRegions(roi_mask, roi_min_area=0, roi_max_area=0):
    """
    Finds connecected component regions in provided roi mask and filters based on area

    Parameters
    -------------
    roi_mask: array of either booleans or ints
        array indicating whether a givin pixel is part of a target object, or a labeled array of target objects
    roi_min_area: int
        minimum required area in number of pixels for an object to be scanned
    roi_max_area: int
        maximum permitted area in number of pixels for an object to be scanned  (0 indicates no max)  

    Returns
    ------------
    dataframe of target regions for the scanning model
    """
    if roi_max_area==0:
        roi_max_area=np.inf
    #if np.max(roi_mask)>1:
    #    section_mask=roi_mask
    #else:
    section_mask=measure.label(roi_mask)
    sections_dict=measure.regionprops_table(section_mask,separator='-',properties=('label','area','bbox'))
    sections_df=pd.DataFrame.from_dict(sections_dict)
    keep_sections=np.logical_and(np.greater_equal(sections_df.loc[:,"area"],roi_min_area),np.less_equal(sections_df.loc[:,"area"],roi_max_area))
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
    for ii,sec in sections.iterrows():
        #print(sec)
        bbox=np.array([sec["bbox-0"],sec["bbox-1"],sec["bbox-2"],sec["bbox-3"]])
        bbox=bbox*2**(roi_mask_ds-scan_ds)
        cur_i=bbox[0]-bbox[0]%scan_step
        
        
        
        while cur_i<bbox[2]:
            #print()
            #print(bbox[3])
            cur_j=bbox[1]-bbox[1]%scan_step
            
            while cur_j<bbox[3]:
                #print(cur_j)
                #print(patch_j)
                if patch_count==batch_size*scan_group_size:
                    batch_df=pd.DataFrame.from_dict({"IPos":patch_i,"JPos":patch_j,"Batch":batch})
                    batch_df.to_csv(f"{image_id}_scan_group_{scan_group_itt}.csv",index=False)
                    patch_count=0
                    batch_itt=0
                    scan_group_itt+=1
                    patch_i=[]
                    patch_j=[]
                    batch=[]

                patch_i.append(int(cur_i))
                patch_j.append(int(cur_j))
                batch.append(batch_itt)
                
                cur_j=cur_j+scan_step
                patch_count+=1
                if patch_count%batch_size==0:
                    batch_itt+=1
            cur_i=cur_i+scan_step
    batch_df=pd.DataFrame.from_dict({"IPos":patch_i,"JPos":patch_j,"Batch":batch})
    batch_df.to_csv(f"{image_id}_scan_group_{scan_group_itt}.csv",index=False)
    return None

def prepForViT(image_id : str, image_dir : str, feature_dir : str, roi_mask_dir : str,
                roi_type : str, roi_min_area : int, roi_max_area: int,
                vit_model_channels : int, scan_step : int, batch_size :int, scan_group_size : int,
                scan_ds : int, roi_mask_ds : int, complete_scan : bool = False, compute_rois : bool = False, 
                roi_identifier_model : str = None, roi_thresh : float = None, scan_mask : bool = False,patch_size:int=None,
                model_checkpoint:str=None,device:str=None, filter_sigma: float=10):
    """
    prepairs for scanning an image with the ViTScanner
    by identifying regions to scan (if nessesary)
    and pre batching the image patches

    Args:
        image_id: id of target slide
        image_dir: path to directory of zarrs converted from slides
        feature_dir: path to directory of zarrs storing the ViT feature representstion
        roi_mask_dir: path to directory of zarrs storing ROI masks of the slides
        roi_type: label indicating which ROI mask should be used to determine the regions to scan
        roi_min_area: minimum required area in number of pixels for an object to be scanned
        roi_max_area: maximum permitted area in number of pixels for an object to be scanned (0 indicates no max)
        vit_model_channels: numper of channels output by the headless ViT
        scan_step: size of step between scan patches
        batch_size: number of patches per batch
        scan_group_size: number of batches to be handeled by each scan process
        scan_ds: number of times to downsample befor scanning (scan patch will be 1/2^(scan_ds) the resolution of the initial image)
        roi_mask_ds: number of times downssampled before identifying the regions to be scanned
        complete_scan: flag indicating if the whole image should be scanned ignoring ROIs
        compute_rois: flag indicating if existing ROIs should be recomputed
        roi_identifier_model: what model to use for identifying ROIs if not precomputed and compute_rois
        roi_thresh: Threshold for simple ROI detection models
        scan_mask : create a mask of which patches have been scanned (useful for sanity checking the scan but creates an ineficent zarr)
        filter_sigma: Sigma value for filter (blur).
    """
    image_path=f"{image_dir}slide_{image_id}.zarr"
    roi_mask_path=f"{roi_mask_dir}roi_masks_{image_id}.zarr"
    feature_path=f"{feature_dir}ViT_features_{image_id}.zarr"

    #pre-create zarr array to contain feature scan (prevents race condition issiues that would arise if initialized at scan time)
    zi=zarr.open(image_path)
    zi=zi["0"]
    zf=zarr.open(feature_path, mode='a')
    zf.zeros(f"Features/{scan_ds}",shape=(vit_model_channels,int(np.ceil(zi[f"{scan_ds}"].shape[3]/scan_step)),int(np.ceil(zi[f"{scan_ds}"].shape[4]/scan_step))),chunks=(vit_model_channels,1,1),overwrite=True)
    if scan_mask:
        zf.zeros(f"ScanMask/{scan_ds}",shape=(int(np.ceil(zi[f"{scan_ds}"].shape[3]/scan_step)),int(np.ceil(zi[f"{scan_ds}"].shape[4]/scan_step))),chunks=(1,1),overwrite=True)

    # compute roi mask if nessasary or indicated by parameters
    zm=zarr.open(roi_mask_path)
    if compute_rois or (f"{roi_type}/{roi_mask_ds}" not in zm):
        if roi_identifier_model=="ChannelMeanThresh":
            id_model=GuasianTissueThresholder(roi_thresh,[0,1,2],lambda x:np.mean(x,axis=0),filter_sigma)
        elif roi_identifier_model=="GreenThresh":
            id_model=GuasianTissueThresholder(roi_thresh,[1],lambda x:x.squeeze(),filter_sigma)
        else:
            raise ValueError("roi_identifier_model must be one of ['ChannelMeanThresh','GreenThresh']")
        image_ds=zarr.open(image_path)[f'0/{roi_mask_ds}']
        roi_mask=id_model(image_ds)
        zm.array(f"{roi_type}/{roi_mask_ds}",roi_mask,overwrite=True)   
    else:
        roi_mask=zm[f"{roi_type}/{roi_mask_ds}"][:,:]
    
    # find roi regions
    if complete_scan:
        sections=pd.DataFrame.from_dict({"bbox-0":[0],"bbox-1":[0],"bbox-2":[zi[f"{scan_ds}"].shape[3]],"bbox-3":[zi[f"{scan_ds}"].shape[4]]})
    else:
        sections=findRegions(roi_mask,roi_min_area,roi_max_area)

    #create scan group csvs
    scanBatching(image_id, sections, scan_step, batch_size, scan_group_size, scan_ds, roi_mask_ds)


if __name__=="__main__":
    CLI(prepForViT)





    



    

