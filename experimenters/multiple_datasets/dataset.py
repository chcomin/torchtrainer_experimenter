from pathlib import Path

import torch
from torchtrainer.datasets.vessel_base import DRIVE, VessMAP
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F


class ValidTransforms:
    """Validation transform that only resizes the image."""

    def __init__(self, resize_size = None, resize_target = True):
        self.resize_size = resize_size
        self.resize_target = resize_target

    def __call__(self, img, target):

        img = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.uint8)
        target = torch.from_numpy(target).unsqueeze(0).to(dtype=torch.uint8)
        
        if self.resize_size is not None:
            img = tv_transf_F.resize(img, self.resize_size)
            if self.resize_target:
                target = tv_transf_F.resize(target, 
                                            self.resize_size, 
                                            interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = img.float()/255
        target = target.to(dtype=torch.int64)[0]

        return img, target

def get_datasets(datasets_root):

    ignore_index = 2  # For the DRIVE dataset
    root_vesshape = Path(datasets_root) / "VessShape"
    root_drive = Path(datasets_root) / "DRIVE"
    root_vessmap = Path(datasets_root) / "VessMAP"

    transforms_256 = ValidTransforms(resize_size=(256, 256))
    transforms_512 = ValidTransforms(resize_size=(512, 512))

    ds_train_vessshape = VessMAP(root_vessmap, keepdim=True, transforms=transforms_256)
    ds_valid_vessshape = VessMAP(root_vessmap, keepdim=True, transforms=transforms_256)
    ds_valid_drive = DRIVE(root_drive, split="test", channels="gray", keepdim=True, 
                     ignore_index=ignore_index, transforms=transforms_512)
    ds_valid_vessmap = VessMAP(root_vessmap, keepdim=True, transforms=transforms_256)

    ds_valids = {
        "VessShape": ds_valid_vessshape,
        "DRIVE": ds_valid_drive,
        "VessMAP": ds_valid_vessmap
    }

    return ds_train_vessshape, ds_valids, ignore_index


