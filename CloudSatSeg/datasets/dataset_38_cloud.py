import os
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, Dataset
import albumentations as albu
from albumentations import pytorch as AT
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from pathlib import Path

import torch


# class CloudDataset(Dataset):
#     def __init__(self, data_dir, b_dir, g_dir, r_dir, nir_dir, gt_dir):
#         print('f')
#         self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
#
#     def __len__(self):
#         print('f')
#
#     def combine_files(self, r_file, g_dir, b_dir, nir_dir, gt_dir):
#         files = {'red': r_file,
#                  'green': g_dir / r_file.name.replace('red', 'green'),
#                  'blue': b_dir / r_file.name.replace('red', 'blue'),
#                  'nir': nir_dir / r_file.name.replace('red', 'nir'),
#                  'gt': gt_dir / r_file.name.replace('red', 'gt')}
#
#         return files

# segmentation dataset
class L8CLoudDataset(Dataset):
    def __init__(self, base_dir, datatype='train', transforms=None, preprocessing=None, include_nir=True):

        if datatype == 'train':
            self.data_dir = os.path.join(base_dir, '38-Cloud_training')
            path_patches = os.path.join(self.data_dir, 'training_patches_38-Cloud.csv')
            label = 'train'
        else:
            self.data_dir = os.path.join(base_dir, '38-Cloud_test')
            path_patches = os.path.join(self.data_dir, 'test_patches_38-Cloud.csv')
            label = 'test'

        self.dict_dir = {}
        list_name = ['blue', 'green', 'red', 'nir', 'gt']
        for cl in list_name:
            self.dict_dir[cl] = os.path.join(self.data_dir, label + '_' + cl)

        self.transforms = transforms
        self.preprocessing = preprocessing
        self.list_patch_names = pd.read_csv(path_patches).iloc[:, 0].tolist()
        self.include_nir=include_nir

    def open_as_array(self, patch_name, include_nir=False, normalize=True):
        list_bands = ['red', 'green', 'blue']
        if include_nir:
            list_bands.append('nir')

        list_array = []
        for band in list_bands:
            path_temp = os.path.join(self.dict_dir[band], band + '_' + patch_name + '.TIF')
            list_array.append(np.array(Image.open(path_temp)))
        raw_rgb = np.stack(list_array, axis=2)
        if normalize:
            raw_rgb = (raw_rgb / np.iinfo(raw_rgb.dtype).max)
        raw_rgb = raw_rgb.astype(np.float32)

        return raw_rgb

    def open_mask(self, patch_name, mask_val=255, add_dims=False):
        band = 'gt'
        path_temp = os.path.join(self.dict_dir[band], band + '_' + patch_name + '.TIF')
        raw_mask = np.array(Image.open(path_temp))
        raw_mask = np.where(raw_mask == mask_val, 1, 0)
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):
        patch_name = self.list_patch_names[idx]
        img = self.open_as_array(patch_name, include_nir=self.include_nir, normalize=True)
        mask = self.open_mask(patch_name, add_dims=False)
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        if self.preprocessing is not None:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        else:
            img = to_tensor(img)
        return img, mask

    def __len__(self):
        return len(self.list_patch_names)

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        # albu.GridDistortion(p=0.5),
        # albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)#,
        # albu.Resize(320, 640)
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640)
    ]
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

def main():
    base_dir = '/dataset/kaggle/38-cloud'
    datatype = 'train'  # 'test'
    include_nir = True
    train_ratio = 0.8

    # set transform
    transforms_train = get_training_augmentation()  #albu.Compose([albu.HorizontalFlip()])
    transforms_test = None #get_validation_augmentation()

    # preprocessing if smp
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn=preprocessing_fn)
    if include_nir:
        preprocessing = None


    dataset = L8CLoudDataset(base_dir=base_dir, datatype=datatype, transforms=transforms_train,
                             preprocessing=preprocessing, include_nir=include_nir)
    img, mask = dataset.__getitem__(100)
    print(img.shape, mask.shape)

    # divide training set and validation set
    n_samples = len(dataset)  # n_samples is 60000
    train_size = int(len(dataset) * train_ratio)  # train_size is 48000
    val_size = n_samples - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(len(train_ds))
    print(len(valid_ds))


if __name__ == "__main__":
    main()
