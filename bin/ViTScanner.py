import zarr
import torch
import torchvision
import pandas as pd
from jsonargparse import CLI
import numpy as np
import warnings
from skimage.filters import gaussian

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
        if len(arr.shape)>3:
            arr_sub=np.copy(arr[0,self.channels,0,:,:])
        else:
            arr_sub=np.copy(arr[self.channels,:,:])
        arr_sub=self.agg_function(arr_sub)
        arr_sub=gaussian(arr_sub,sigma=self.sigma)
        if np.max(arr_sub)>1:
            arr_sub=arr_sub/255
        return np.less_equal(arr_sub,self.thresh)




#To-Do figure out how to set the arcetecture via string
def vitScan(image_id : str, scan_group_csv : str, image_dir : str, feature_dir : str, scan_step : int, scan_ds : int, patch_size : int, model_checkpoint : str = None, device : str = 'auto', scan_mask : bool = False,
            roi_mask_dir : str=None,
            roi_type : str=None, roi_min_area : int=None, roi_max_area: int=None,
            vit_model_channels : int=None, batch_size :int=None, scan_group_size : int=None,
            roi_mask_ds : int=None, complete_scan : bool = False, compute_rois : bool = False, 
                roi_identifier_model : str = None, roi_thresh : float = None):
    """
    Scans selected patches of an image using a pretrained pytorch model and saves features to a prepaired zarr file. 

    Parameters
    -------------
    image_id : id of target slide
    scan_group_csv : path to csv containing patch coordinates for scanning
    image_dir : path to directory of zarrs converted from slides
    feature_dir : path to directory of zarrs storing the ViT feature representstion
    scan_step : size of step between patches
    scan_ds : number of times to downsample before scanning (scan patch will be 1/2^(scan_ds) the resolution of the initial image)
    patch_size : size of input to the model
    model_checpoint : path to .pth file with alternate weights
    device : target torch device, must be one of ['auto', 'cuda', 'cpu']
    scan_mask : create a mask of which patches have been scanned (useful for sanity checking the scan but creates an ineficent zarr)
    """
    if roi_identifier_model=="ChannelMeanThresh":
            roi_id_model=GuasianTissueThresholder(roi_thresh,[0,1,2],lambda x:np.mean(x,axis=0),10)
    elif roi_identifier_model=="GreenThresh":
            roi_id_model=GuasianTissueThresholder(roi_thresh,[1],lambda x:x.squeeze(),10)
    else:
        raise ValueError("roi_identifier_model must be one of ['ChannelMeanThresh','GreenThresh']")
    
    if device=="auto":
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"
    elif device=="cuda":
        if not torch.cuda.is_available():
            warnings.warn("GPU unavailable, using CPU")
            device="cpu"
    elif device!="cpu":
        raise ValueError("device must be one of ['auto', 'cuda', 'cpu']")

    model=torchvision.models.vit_h_14(image_size=patch_size)

    
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))
    model.heads=torch.nn.Sequential(torch.nn.Identity())
    model.to(device)

    image_path=f"{image_dir}slide_{image_id}.zarr"
    feature_path=f"{feature_dir}ViT_features_{image_id}.zarr"
    zi=zarr.open(image_path,mode='r+')
    image=zi[f"0/{scan_ds}"]
    zf=zarr.open(feature_path)
    scan_groups_df=pd.read_csv(scan_group_csv)
    scan_groups=scan_groups_df.groupby("Batch")
    for _,scan_group in scan_groups:
        
        batch=np.zeros((scan_group.shape[0],3,patch_size,patch_size))
        keep_patches=np.array([],dtype=np.uint64)
        for ii, row in enumerate(scan_group.itertuples()):
            pad=False
            if row.IPos>(image.shape[3]-patch_size):
                h=image.shape[3]-row.IPos
                pad=True
            else:
                h=patch_size
            if row.JPos>(image.shape[4]-patch_size):
                w=image.shape[4]-row.JPos
                pad=True
            else:
                w=patch_size
            patch=np.copy(image[0,:,0,row.IPos:(row.IPos+h),row.JPos:(row.JPos+w)])
            if pad:
                patch=np.pad(patch,((0,0),(0,patch_size-h),(0,patch_size-w)),constant_values=255)
            patch_roi=roi_id_model(patch)
            if (np.sum(patch_roi)/np.prod(patch_roi.shape))>0.7:
                keep_patches=np.append(keep_patches,ii)
                batch[ii,:,:,:]=np.copy(patch)
            batch=batch[keep_patches.astype(np.uint64),:,:,:]
        batch=torch.Tensor(batch,device=device)
        with torch.inference_mode():
            features=model(batch)
        features=features.detach()
        for ii, row in enumerate(scan_group.itertuples()):
            if ii in keep_patches:
                batch_idx=np.equal(ii,keep_patches)
            else:
                continue
            ipos=int(row.IPos/scan_step)
            jpos=int(row.JPos/scan_step)
            zf[f"Features/{scan_ds}"][:,ipos,jpos]=features[batch_idx,:].numpy()
            if scan_mask:
                zf[f"ScanMask/{scan_ds}"][ipos,jpos]=1
    return None


    

if __name__=="__main__":
    CLI(vitScan)
