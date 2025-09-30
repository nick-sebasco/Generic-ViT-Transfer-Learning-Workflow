import warnings
import numpy as np
import pandas as pd
import torch
import torchvision
import zarr
from jsonargparse import CLI


from stain_normalize import NormContext


# -------------------------
# local helpers
# -------------------------
def _ensure_uint8(chw: np.ndarray) -> np.ndarray:
    if chw.dtype == np.uint8:
        return chw
    out = chw.astype(np.float32)
    for c in range(out.shape[0]):
        mn, mx = float(out[c].min()), float(out[c].max())
        if mx <= mn:
            out[c] = 0
        else:
            out[c] = (out[c] - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# vitScan
# -------------------------
def vitScan(
    image_id: str,
    scan_group_csv: str,
    image_dir: str,
    feature_dir: str,
    roi_mask_dir: str,
    vit_model_channels: int,
    scan_step: int,
    scan_ds: int,
    patch_size: int,
    batch_size: int = 10,
    scan_group_size: int = 3,
    model_checkpoint: str = "",
    device: str = "auto",
    roi_type: str = "GreenThresh",
    roi_mask_ds: int = 4,
    scan_mask: bool = True,
    apply_normalization: bool = False,
    normalization_method: str = "reinhard",
    normalization_ref_zarr: str = "",
    normalization_ref_roi_zarr: str = "",
    normalization_ref_ds: str = None,
    normalization_ref_roi_type: str = None,
    normalization_ref_roi_ds: int = None,
    normalization_ref_size: int = 512,
    normalization_ref_strategy: str = "random",
    feature_tag: str = "",
) -> int:
    """
    Scan (feature-extract) ViT patches for a slide, optionally applying stain
        normalization per patch.

    Parameters
    ----------
    image_id : str
        Slide ID (basename). Image Zarr is expected at: {image_dir}{image_id}.zarr
    scan_group_csv : str
        Path to a single scan group CSV produced by ScanPrep (contains IPos, JPos, Batch).
    image_dir : str
        Directory containing slide image zarrs (pyramids in OME-like [T,C,Z,Y,X]).
    feature_dir : str
        Directory for feature zarrs; features are written to ViT_features_{image_id}{feature_tag}.zarr.
    roi_mask_dir : str
        Directory of ROI mask zarrs (e.g., roi_masks_{image_id}.zarr).
    vit_model_channels : int
        Output channels of the headless ViT (pre-created by ScanPrep in the feature zarr).
    scan_step : int
        Step size in pixels between patches at the scan_ds resolution.
    scan_ds : int
        Log2 downsample level used for scanning (read from "0/{scan_ds}" in the image zarr).
    patch_size : int
        Input patch size for the ViT model (pixels at "0/{scan_ds}").
    batch_size : int
        Patches per batch (used by ScanPrep for CSV grouping; not re-batching here).
    scan_group_size : int
        Batches per scan group (informational; grouping is defined by the CSV).
    model_checkpoint : str
        Optional path to a .pth checkpoint to load into the ViT backbone.
    device : str
        'auto', 'cuda', or 'cpu'.
    roi_type : str
        ROI group name within ROI mask zarr (e.g., "GreenThresh").
    roi_mask_ds : int
        Log2 downsample level in the ROI zarr for the selected ROI mask.
    scan_mask : bool
        If True, mark visited patch positions in ScanMask/{scan_ds} within the feature zarr.
    apply_normalization : bool
        If True, apply stain normalization per kept patch before feeding the model.
    normalization_method : str
        Currently supports "reinhard".
    normalization_ref_zarr : str
        Reference slide zarr (used to fit the normalizer).
    normalization_ref_roi_zarr : str
        Reference slide ROI zarr (used to sample tissue-only reference patch).
    normalization_ref_ds : str
        Dataset key for the reference image (e.g., "0/6"). Defaults to f"0/{scan_ds}" if None.
    normalization_ref_roi_type : str
        ROI type for the reference ROI zarr; defaults to `roi_type` if None.
    normalization_ref_roi_ds : int
        ROI ds level for the reference ROI zarr; defaults to `roi_mask_ds` if None.
    normalization_ref_size : int
        Reference patch size (pixels).
    normalization_ref_strategy : str
        "random" or "median" for reference patch location strategy.
    feature_tag : str
        Optional suffix for feature zarr name (e.g., "_reinhard") to avoid overwriting raw features.

    Notes
    -----
    - Patches are kept only if ≥ 75% tissue overlap with the ROI mask window.
    - Stain normalization, if enabled, is applied per kept patch in-memory (no normalized images are written).
    - Features are written to: ViT_features_{image_id}{feature_tag}.zarr/Features/{scan_ds} at [C, I, J].
    """
    # ------------- device selection -------------
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("GPU unavailable, falling back to CPU.")
            device = "cpu"
    elif device != "cpu":
        raise ValueError("device must be one of ['auto', 'cuda', 'cpu']")

    # ------------- model -------------
    model = torchvision.models.vit_h_14(image_size=patch_size)
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.heads = torch.nn.Sequential(torch.nn.Identity())
    model.to(device)
    model.eval()

    # ------------- paths -------------
    image_path   = f"{image_dir}{image_id}.zarr"
    # Added feature tag, so when you run with a particular normalization method
    # the old features are not overwritten.
    feature_path = f"{feature_dir}ViT_features_{image_id}{feature_tag}.zarr"
    roi_path     = f"{roi_mask_dir}roi_masks_{image_id}.zarr"

    # ------------- open zarrs -------------
    zi = zarr.open(image_path, mode="r")
    # expected shape [T,C,Z,Y,X]
    image = zi[f"0/{scan_ds}"]

    # precreated by ScanPrep
    zf = zarr.open(feature_path, mode="r+")
    zm = zarr.open(roi_path, mode="r")
    # 2D mask
    roi_mask = zm[f"{roi_type}/{roi_mask_ds}"]
    scale_dif = 2 ** (roi_mask_ds - scan_ds)

    # ------------- build normalization context once -------------
    ref_ds_key = normalization_ref_ds or f"0/{scan_ds}"
    ref_roi_type = normalization_ref_roi_type or roi_type
    ref_roi_ds = (
        normalization_ref_roi_ds
        if normalization_ref_roi_ds
        is not None else roi_mask_ds
    )
    ref_roi_key = f"{ref_roi_type}/{ref_roi_ds}"

    norm_ctx = NormContext.build(
        apply_normalization=apply_normalization,
        method=normalization_method,
        ref_zarr=normalization_ref_zarr,
        ref_roi_zarr=normalization_ref_roi_zarr,
        ref_ds_key=ref_ds_key,
        ref_roi_key=ref_roi_key,
        ref_size=normalization_ref_size,
        ref_strategy=normalization_ref_strategy,
    )

    # attrs on feature zarr
    for k, v in norm_ctx.meta.items():
        zf.attrs[k] = v
    zf.attrs["feature_tag"] = feature_tag

    # ------------- load scan groups -------------
    scan_groups_df = pd.read_csv(scan_group_csv)
    scan_groups = scan_groups_df.groupby("Batch")

    # ------------- main loop -------------
    for _, scan_group in scan_groups:
        print(f"Scanning group size: {scan_group.shape[0]}")
        # pre-allocate; we’ll subset down to kept patches
        batch = np.zeros((scan_group.shape[0], 3, patch_size, patch_size), dtype=np.uint8)
        keep_patches = np.array([], dtype=np.uint64)

        for ii, row in enumerate(scan_group.itertuples()):
            pad = False
            if row.IPos > (image.shape[3] - patch_size):
                h = image.shape[3] - row.IPos
                pad = True
            else:
                h = patch_size
            if row.JPos > (image.shape[4] - patch_size):
                w = image.shape[4] - row.JPos
                pad = True
            else:
                w = patch_size

            # CHW patch from pyramid
            patch = np.copy(image[0, :, 0, row.IPos:(row.IPos + h), row.JPos:(row.JPos + w)])

            # corresponding ROI window (in ROI space)
            r0 = int(np.round(row.IPos / scale_dif))
            r1 = int(np.round((row.IPos + h) / scale_dif))
            c0 = int(np.round(row.JPos / scale_dif))
            c1 = int(np.round((row.JPos + w) / scale_dif))
            patch_roi = roi_mask[r0:r1, c0:c1]

            if pad:
                patch = np.pad(patch, ((0, 0), (0, patch_size - h), (0, patch_size - w)), constant_values=255)
                pad_r = int(np.round((patch_size - h) / scale_dif))
                pad_c = int(np.round((patch_size - w) / scale_dif))
                patch_roi = np.pad(patch_roi, ((0, max(0, pad_r)), (0, max(0, pad_c))), constant_values=0)

            # keep only if ≥75% tissue
            if (np.sum(patch_roi) / np.prod(patch_roi.shape)) > 0.75:
                # normalize (if enabled)
                patch_u8 = _ensure_uint8(patch)
                patch_u8 = norm_ctx.normalize_patch(patch_u8)  # CHW uint8 -> CHW uint8 (or no-op)

                keep_patches = np.append(keep_patches, ii)
                batch[ii, :, :, :] = patch_u8

        # nothing to do for this group
        if keep_patches.size == 0:
            continue

        # subset to kept patches
        batch = batch[keep_patches.astype(np.uint64), :, :, :]

        # to torch float32 in [0,1]
        batch_t = torch.tensor(batch, device=device, dtype=torch.float32) / 255.0

        with torch.inference_mode():
            features_t = model(batch_t)

        features = features_t.detach().cpu().numpy()

        # write each kept patch to feature zarr
        for ii, row in enumerate(scan_group.itertuples()):
            if ii in keep_patches:
                batch_idx = np.equal(ii, keep_patches)
            else:
                continue
            ipos = int(row.IPos / scan_step)
            jpos = int(row.JPos / scan_step)
            zf[f"Features/{scan_ds}"][:, ipos, jpos] = np.squeeze(
                features[batch_idx, :]
            )
            if scan_mask:
                zf[f"ScanMask/{scan_ds}"][ipos, jpos] = 1

    return 0


#To-Do figure out how to set the arcetecture via string
def vitScan_V0(image_id : str, scan_group_csv : str, image_dir : str, feature_dir : str, scan_step : int, scan_ds : int, patch_size : int, model_checkpoint : str = None, device : str = 'auto', scan_mask : bool = False,
            roi_mask_dir : str=None,
            roi_type : str=None, roi_min_area : int=None, roi_max_area: int=None,
            vit_model_channels : int=None, batch_size :int=None, scan_group_size : int=None,
            roi_mask_ds : int=None, complete_scan : bool = False, compute_rois : bool = False, 
                roi_identifier_model : str = None, roi_thresh : float = None, filter_sigma: float = None):
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
    roi_path=f"{roi_mask_dir}roi_masks_{image_id}.zarr"
    zi=zarr.open(image_path,mode='r+')
    image=zi[f"0/{scan_ds}"]
    zf=zarr.open(feature_path)
    zm=zarr.open(roi_path)
    roi_mask=zm[f"{roi_type}/{roi_mask_ds}"]
    scale_dif=2**(roi_mask_ds-scan_ds)
    scan_groups_df=pd.read_csv(scan_group_csv)
    scan_groups=scan_groups_df.groupby("Batch")
    for _,scan_group in scan_groups:
        print()
        print(scan_group.shape[0])
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
            patch_roi=roi_mask[int(np.round(row.IPos/scale_dif)):int(np.round((row.IPos+h)/scale_dif)),int(np.round(row.JPos/scale_dif)):int(np.round((row.JPos+w)/scale_dif))]
            
            if pad:
                patch=np.pad(patch,((0,0),(0,patch_size-h),(0,patch_size-w)),constant_values=255)
                patch_roi=np.pad(patch_roi,((0,int(np.round((patch_size-h)/scale_dif))),(0,int(np.round((patch_size-w)/scale_dif)))),constant_values=255)
            if (np.sum(patch_roi)/np.prod(patch_roi.shape))>0.75:
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
            zf[f"Features/{scan_ds}"][:,ipos,jpos]=np.squeeze(features[batch_idx,:].numpy())
            if scan_mask:
                zf[f"ScanMask/{scan_ds}"][ipos,jpos]=1
    return 0


if __name__=="__main__":
    CLI(vitScan)

