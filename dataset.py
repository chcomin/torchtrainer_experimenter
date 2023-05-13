from functools import partial
import numpy as np
from PIL import Image
import torch
import albumentations as aug
from albumentations.pytorch import ToTensorV2
from torchtrainer.imagedataset import ImageSegmentationDataset

def name_to_label_map(img_path):
    return img_path.replace('.tiff', '.png')

def img_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil)

def label_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil)//255

def transform(image, mask, transform_comp):
    """Given an image and a mask, apply transforms in `transform_comp`"""
    res = transform_comp(image=image, mask=mask)
    image, mask = res['image'], res['mask']

    return image, mask.long()

def create_transform(mean, std, crop_size=None, use_simple=True):
    """Create a transform function with signature transform(image, label)."""

    def generate_multiplicative(mult_val_limit):
        """Generate custom function for Albumentations. The augmentation multiply
        the image by a constant value drawn uniformly in the range defined by `mult_val_limit`."""
        def custom(img, **kwargs):
            mult_val = np.random.rand()*(mult_val_limit[1]-mult_val_limit[0]) + mult_val_limit[0]
            return (img*mult_val).astype(np.uint8)
        return custom
    
    if crop_size is None:
        random_crop = aug.NoOp(p=1.)
    else:
        random_crop = aug.RandomCrop(crop_size[0], crop_size[1])
        
    if use_simple:
        transform_comp = aug.Compose([
            random_crop,
            aug.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        transform_comp = aug.Compose([
            random_crop,
            aug.OneOf([
                aug.GaussianBlur(blur_limit=(3, 7)),
                aug.Sharpen(alpha=(0.2, 0.5), lightness=(0., 1.)),
                aug.UnsharpMask(blur_limit=(3, 15), alpha=(0.4, 1.), threshold=0, p=0.1)  # slow, 0.038s/img
            ], p=1.),
            aug.OneOf([
                aug.RandomGamma(gamma_limit=(90, 150)),
                aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(-0.2, 0.5)),
                aug.Lambda(name='multiply', image=generate_multiplicative((0.7, 1.)), p=0.1)
            ], p=1.),
            aug.GaussNoise(var_limit=(50, 250), p=1.),  # slow, 0.051s/img
            aug.Flip() ,
            aug.RandomRotate90(),
            aug.Transpose(),
            aug.ShiftScaleRotate(shift_limit_x=0.1, shift_limit_y=0.1, scale_limit=0.25, rotate_limit=45),
            aug.CLAHE(clip_limit=(1., 2.), tile_grid_size=(16, 16), p=0.1),
            aug.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    transform_func = partial(transform, transform_comp=transform_comp)

    return transform_func

def create_datasets(img_dir, label_dir, crop_size=None, train_val_split=0.1, use_simple=True, seed=None):
    """Create dataset from directory. 
    
    Args
    crop_size: (height,width) to crop the images
    train_val_split: percentage of images used for validation
    use_simple: if True, use only crop, normalization and ToTensor. If False, use many data augmentation
    transformations.
    seed: seed for splitting the data
    """

    mean_data = 0.1373
    std_data = 0.0482
    class_weights = torch.tensor([0.367, 0.633])  # On average, 63.3% of the images is background

    train_transform = create_transform(mean=mean_data, std=std_data, crop_size=crop_size, use_simple=use_simple)
    valid_transform = create_transform(mean=mean_data, std=std_data, crop_size=None, use_simple=True)

    ds = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener, label_opener=label_opener)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)

    meta = {
        'mean_data':mean_data,
        'std_data':std_data,
        'class_weights':class_weights
    }

    return ds_train, ds_valid, meta