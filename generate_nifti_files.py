""" You need to use python: 3.9.20 to run this script, check labelSys requirements and packages for more details."""

import json
from typing import Tuple, Union
from labelSys.utils.labelReaderV2 import LabelSysReader
from monsoonToolBox.arraytools.img2d import Img2D
from monsoonToolBox.filetools import *
import skimage.transform
import cv2 as cv
import numpy as np
from progress.bar import Bar
import pickle
import nibabel as nib
import os
from tqdm.auto import tqdm
from nnunet.utilities.file_conversions import convert_3d_tiff_to_nifti
import skimage.transform, tifffile
from utils.npImgTools import stretchArr
from batchgenerators.utilities.file_and_folder_operations import *

OUT_DIR = "output_nifti"
OUT_DIR_SEG = "output_nifti_seg"
PROJECT_NAME = "TMJ Test"
IM_HEIGHT = 256
IM_WIDTH = 256
INPUT_DIR = "data"
LBL_NUM = {"Disc": 1, "Condyle": 2, "Eminence": 3}
SEQ_ARR_T = Union[List[np.ndarray], np.ndarray]  

""" Reference: https://github.com/aswahd/TMJ-Disk-Dislocation-Classification/blob/main/nnUNetTMJ/datasetConversion/prepareDatasets_TMJ.py"""
def _resize(images: SEQ_ARR_T, masks: SEQ_ARR_T) -> Tuple[SEQ_ARR_T, SEQ_ARR_T]:
    """Resize images and masks

    Args:
            images (np.ndarray): array of 2D images
            masks (np.ndarray): array of 2D masks

    Returns:
            (images_resized, masks_resized)
    """
    images = list(
        map(
            lambda x: skimage.transform.resize(x, (IM_HEIGHT, IM_WIDTH), order=1),
            images,
        )
    )
    masks = list(
        map(
            lambda x: cv.resize(
                x.round().astype(np.uint8),
                (IM_HEIGHT, IM_WIDTH),
                interpolation=cv.INTER_NEAREST,
            ),
            masks,
        )
    )
    return images, masks

""" Reference: https://github.com/aswahd/TMJ-Disk-Dislocation-Classification/blob/main/nnUNetTMJ/datasetConversion/prepareDatasets_TMJ.py"""
def generateData(
    ori_data_path: str, out_img_dir: str, out_seg_dir: str, unique_name: str
):
    img_tiff_path = f"{out_img_dir}/temp_img.tif"
    msk_tiff_path = f"{out_seg_dir}/temp_msk.tif"
    
    legacy_config = '\
		{"labels": ["Disc", "Condyle", "Eminence"], "label_modes": [0, 1, 1], "label_colors": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], "label_steps": [8, 30, 15], "loading_mode": 0, "default_series": "SAG PD", "2D_magnification": 1, "max_im_height": 512}\
		'
    legacy_config = json.loads(legacy_config)
    reader = LabelSysReader([ori_data_path], legacy_config)
    data = reader[0]
    imgs = data.images
    msks = [data.grayMask(i, LBL_NUM) for i in range(len(data.masks))]
    header = data.header
    imgs, msks = _resize(imgs, msks)
    imgs = stretchArr(np.array(imgs), max_val=1)
    spacing = header["Spacing"]
    tifffile.imsave(img_tiff_path, imgs)
    tifffile.imsave(msk_tiff_path, msks)
    
    output_img_file = join(out_img_dir, unique_name)
    output_seg_file = join(out_seg_dir, unique_name)

    convert_3d_tiff_to_nifti(
        [img_tiff_path], output_img_file, spacing=spacing, is_seg=False
    )
    convert_3d_tiff_to_nifti(
        [msk_tiff_path], output_seg_file, spacing=spacing, is_seg=True
    )
  

def main():
    print("Generating NIfTI files...")

    maybe_mkdir_p(OUT_DIR)
    maybe_mkdir_p(OUT_DIR_SEG)

    subdirs = subDirs(INPUT_DIR)

    case_mapping = {}
    bar = Bar("Processing cases", max=len(subdirs))

    for idx, ori_path in enumerate(subdirs):
        case_id = f"case{idx:04d}_0000"  
        original_name = os.path.basename(ori_path.rstrip("/"))

        generateData(
            ori_data_path=ori_path,
            out_img_dir=OUT_DIR,
            out_seg_dir=OUT_DIR_SEG,
            unique_name=case_id,
        )

        case_mapping[case_id] = original_name
        bar.next()

    bar.finish()
    
    mapping_path = os.path.join(OUT_DIR, "case_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(case_mapping, f, indent=4)

    print("Images and masks saved as NIfTI files.")
    print("Images located at:", OUT_DIR)
    print("Masks located at:", OUT_DIR_SEG)
    print("Mapping saved at:", mapping_path)
    
    
if __name__ == "__main__":
    main()

