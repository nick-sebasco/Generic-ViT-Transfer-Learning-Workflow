import imageio as iio
import zarr
import argparse
import numpy as np
import re

def qupath2zarrmask(file_name,ome_tif_dir,zarr_dir,downscale,mask_names):
    if downscale<1:
        downscale=1/downscale
    downscale=np.log2(downscale)
    if downscale%1!=0:
        raise ValueError("downscale must be a power of 2 or 1 divided by a power of 2")
    else:
        downscale=int(downscale)
    image_id=re.sub(r"\.ndpi.+","",file_name)
    image_id=re.sub('[ -,]','_',image_id)
    img=iio.imread(f"{ome_tif_dir}{file_name}")
    z=zarr.open(f"{zarr_dir}roi_masks_{image_id}.zarr",mode='a')
    for ii, name in enumerate(mask_names):
        z.create_array(f"{name}/{downscale}",data=img[:,:,ii])

CLI=argparse.ArgumentParser()
CLI.add_argument("file_name",type=str)
CLI.add_argument("ome_tif_dir",type=str)
CLI.add_argument("zarr_dir",type=str)
CLI.add_argument('downscale',type=int)
CLI.add_argument('mask_names',type=str)

args=CLI.parse_args()
mask_name_list=args.mask_names.replace(" ","").split(",")
qupath2zarrmask(args.file_name, args.ome_tif_dir,args.zarr_dir,args.downscale,mask_name_list)