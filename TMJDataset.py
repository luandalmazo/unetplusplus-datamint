import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import cv2
import os
import nibabel as nib
import torchvision.transforms.functional as torchTF
from torchvision.transforms.functional import InterpolationMode

from datamint import Api

PROJECT_NAME = "TMJ Test"
NUM_CLASSES = 4

api = Api()

class TMJDataset2D(Dataset):
    def __init__(
        self,
        split: str | None = None,
        use_augmentation: bool = True,
    ):
        super().__init__()
        self._mask_tmp_dir = os.path.join(
            os.environ.get("SCRATCH", "/tmp"),
            "tmj_mask_cache"
        )
        os.makedirs(self._mask_tmp_dir, exist_ok=True)
        self.num_classes = NUM_CLASSES
        self.augmentation = use_augmentation and (split == "train")

        ''' Data Loading from Datamint '''
        self.resources = list(  
            api.resources.get_list(
                project_name=PROJECT_NAME,
                tags=[f"split:{split}"] if split else None,
            )
        )
        
        self._resource_map = {r.id: r for r in self.resources}

        ''' Load all segmentation annotations for the resources '''
        all_annotations = api.annotations.get_list(
            resource=self.resources,
            annotation_type="segmentation"
        )

        self.resource_annotations = defaultdict(list)
        
        for ann in all_annotations:
            self.resource_annotations[ann.resource_id].append(ann)
            
        self.slice_index = []
        self._volume_cache = {}
        self._mask_volume_cache = {}

        for resource in self.resources:
            vol = resource.fetch_file_data(auto_convert=True, use_cache=False)
            vol_np = vol.get_fdata() if hasattr(vol, "get_fdata") else vol.pixel_array

            if vol_np.ndim != 3:
                raise RuntimeError(
                    f"Resource {resource.filename} is not 3D (shape={vol_np.shape})"
                )

            num_slices = vol_np.shape[0]
            self._volume_cache[resource.id] = vol_np

            for z in range(num_slices):
                self.slice_index.append((resource.id, z))

        for resource_id, anns in self.resource_annotations.items():
            self._mask_volume_cache[resource_id] = []

            for ann in anns:
                try:
                    raw = ann.fetch_file_data(auto_convert=True, use_cache=True)

                    if isinstance(raw, (bytes, bytearray)):
                        tmp_path = os.path.join(
                            self._mask_tmp_dir,
                            f"{ann.id}.nii.gz"
                        )

                        if not os.path.exists(tmp_path):
                            with open(tmp_path, "wb") as f:
                                f.write(raw)

                        nii = nib.load(tmp_path)
                        data = nii.dataobj  
                        vol = np.asarray(data)                        

                    elif hasattr(raw, "get_fdata"):
                        vol = raw.get_fdata()
                    else:
                        vol = np.asarray(raw)

                    vol = np.asarray(vol, dtype=np.int64)
                    self._mask_volume_cache[resource_id].append(vol)

                except Exception as e:
                    print(
                        f"[WARN][MASK CACHE FAIL] "
                        f"resource={resource_id} ann={ann.id} err={e}"
                    )

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        resource_id, slice_idx = self.slice_index[idx]

        ''' Get resource and cached volume '''
        resource = self._resource_map[resource_id]
        vol = self._volume_cache[resource_id]

        image = vol[slice_idx, :, :].astype(np.float32)

        original_height, original_width = image.shape

        ''' Load segmentation masks (union if multiple) '''
        cached_masks = self._mask_volume_cache.get(resource_id, [])

        if not cached_masks:
            mask = np.zeros((original_height, original_width), dtype=np.int64)
        else:
            masks = []

            for vol in cached_masks:
                if vol.ndim == 3:
                    masks.append(vol[slice_idx])
                elif vol.ndim == 2:
                    masks.append(vol)

            if len(masks) == 0:
                mask = np.zeros((original_height, original_width), dtype=np.int64)
            else:
                mask = np.maximum.reduce(masks)

        
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (256, 256), interpolation=cv2.INTER_NEAREST)
        
        ''' Normalization '''
        vmin = np.percentile(image, 1)
        vmax = np.percentile(image, 99)
        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin + 1e-8)

    
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        mask = torch.from_numpy(mask).long()                  # (H, W)

        ''' Data Augmentation '''
        if self.augmentation:
            image, mask = self.randomTransform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "filename": f"{resource.filename}_z{slice_idx:03d}",
        }

    # ---------------------------------------------------------------------------------------------------------------------------------
    """
    Taken from: https://github.com/aswahd/TMJ-Disk-Dislocation-Classification/blob/main/UNetPPTMJ/dataloading/TMJDataset.py
    Apply random transformations to the image and label.
    """
    def randomTransform(self, img, lbl):
        _random_transforms = [
            self._randomAdjustBright,
            self._randomRotate,
            self._randomAffine,
            self._flipLeftRight,
        ]
        n_transforms = random.randint(0, 2)
        transforms_list = random.sample(_random_transforms, n_transforms)

        for _transform in transforms_list:
            if random.random() > 0.5:
                img, lbl = _transform(img, lbl)
        return img, lbl

    def _randomRotate(self, img, lbl, degree=30):
        angle = random.randint(-degree, degree)
        img = torchTF.rotate(img, angle)
        lbl = torchTF.rotate(
            lbl.unsqueeze(0),
            angle,
            interpolation=InterpolationMode.NEAREST
        )
        return img, lbl[0]

    def _randomAdjustBright(self, img, lbl):
        factor = 2 * random.random() - 1
        factor = (1 / 4) ** factor
        img = torchTF.adjust_brightness(img, factor)
        return img, lbl

    def _randomAffine(self, img, lbl):
        shear = random.random() * 15
        scale = random.random() * 0.25 + 1
        trans_x = np.random.randint(-50, 50)
        trans_y = np.random.randint(-50, 50)
        angle = random.randint(-10, 10)

        img = torchTF.affine(
            img,
            angle=angle,
            translate=(trans_x, trans_y),
            shear=shear,
            scale=scale,
        )
        lbl = torchTF.affine(
            lbl.unsqueeze(0),
            angle=angle,
            translate=(trans_x, trans_y),
            shear=shear,
            scale=scale,
            interpolation=InterpolationMode.NEAREST
        )
        return img, lbl[0]

    def _flipLeftRight(self, img, lbl):
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-1])
            lbl = torch.flip(lbl, dims=[-1])
        return img, lbl
