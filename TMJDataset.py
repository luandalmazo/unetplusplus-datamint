import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import cv2

import torchvision.transforms.functional as torchTF
from torchvision.transforms.functional import InterpolationMode

from datamint import Api

PROJECT_NAME = "TMJ Study"
NUM_CLASSES = 4

api = Api()

class TMJDataset2D(Dataset):
    def __init__(
        self,
        split: str | None = None,
        use_augmentation: bool = True,
    ):
        super().__init__()

        self.num_classes = NUM_CLASSES
        self.augmentation = use_augmentation and (split == "train")

        ''' Data Loading from Datamint '''
        self.resources = list(
            api.resources.get_list(
                project_name=PROJECT_NAME,
                tags=[f"split:{split}"] if split else None,
            )
        )

        ''' Load all segmentation annotations for the resources '''
        all_annotations = api.annotations.get_list(
            resource=self.resources,
            annotation_type="segmentation"
        )

        self.resource_annotations = defaultdict(list)
        for ann in all_annotations:
            self.resource_annotations[ann.resource_id].append(ann)

    def __len__(self):
        return len(self.resources)

    def __getitem__(self, idx):
        resource = self.resources[idx]

        ''' Load image (PIL -> NumPy) '''
        image = resource.fetch_file_data(
            auto_convert=True,
            use_cache=True
        )  

        original_height, original_width = image.Rows, image.Columns

        image = image.pixel_array.astype(np.float32)
        
                
        ''' Load segmentation masks (union if multiple) '''
        anns = self.resource_annotations[resource.id]
        no_mask = False
        
        if not anns:
            mask = np.zeros((original_height, original_width), dtype=np.int64)
        else:            
            masks = []
            for ann in anns:
        
                mask_img = ann.fetch_file_data(
                    auto_convert=True,
                    use_cache=True
                )  

                mask = np.array(mask_img).astype(np.int64)

                masks.append(mask)

            ''' Combine multiple masks into a single mask '''
            mask = np.maximum.reduce(masks)

        ''' Normalization '''
        vmin = np.percentile(image, 1)
        vmax = np.percentile(image, 99)
        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin + 1e-8)

        ''' Convert to torch tensors '''
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        mask = torch.from_numpy(mask).long()                  # (H, W)

        ''' Data Augmentation '''
        if self.augmentation:
            image, mask = self.randomTransform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "filename": resource.filename,
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
