import os
import cv2
from tqdm import tqdm

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

from utils.process import imap_unordered_bar, argwrapper


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
    def __init__(self, base_dir, datatype='train', transforms=None, preprocessing=None, include_nir=True,
                 non_null_rate=None, cloud_rate=None, processes=1):
        """ get torch datasets

        Args:
            base_dir (str): path to dataset
            datatype (str): 'train'-> training data, other -> test data
            transforms (): torch transform
            preprocessing (): preprocessing function
            include_nir (): if True, include nir. If not, only RGB
            non_null_rate (float or None): threshold of null pixel rate for picked image (0.0 or None: all, 1.0:no null)
            cloud_rate (float or None): threshold of cloud rate for picked image (None: all, 0.0: include non cloud image, 1.0: only all clouded image)
            processes (int): number of threads for calculating cloud rate in each images
        """

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
        self.include_nir = include_nir

        if (non_null_rate is not None) or (cloud_rate is not None):
            df = self.get_df_ratio(processes=processes)
            if non_null_rate is not None:
                sr_bool = df.loc[:, 'non_null_rate'] >= non_null_rate
            else:
                sr_bool = df.loc[:, 'non_null_rate'] >= 0.0

            if cloud_rate is not None:
                sr_bool = sr_bool & (df.loc[:, 'cloud_rate'] >= cloud_rate)
            self.list_patch_names = df.loc[sr_bool, 'patch_name'].tolist()

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

    def open_url(self, patch_name, datatype='train'):
        if datatype == 'train':
            list_bands = ['red', 'green', 'blue', 'nir']
            list_url = []
            for band in list_bands:
                list_url.append(os.path.join(self.dict_dir[band], band + '_' + patch_name + '.TIF'))
            return list_url
        else:
            band = 'gt'
            return os.path.join(self.dict_dir[band], band + '_' + patch_name + '.TIF')

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

    def __geturl__(self, idx):
        patch_name = self.list_patch_names[idx]
        img_url = self.open_url(patch_name=patch_name, datatype='train')
        mask_url = self.open_url(patch_name=patch_name, datatype='val')

        return img_url, mask_url

    def get_image_type(self, patch_name):
        # patch_name = self.list_patch_names[idx]
        img = self.open_as_array(patch_name, include_nir=self.include_nir, normalize=False)
        mask = self.open_mask(patch_name, add_dims=False)
        list_ratio = []
        for band in range(img.shape[2]):
            list_ratio.append(np.sum(img[:, :, band] != 0) / (img.shape[0] * img.shape[1]))
        ratio = np.min(list_ratio)

        # cloud rate
        index_val = np.where(img[:, :, 0] != 0)
        cloud_rate = mask[np.where(img[:, :, 0] != 0)].sum() / mask[np.where(img[:, :, 0] != 0)].shape[0]

        return patch_name, ratio, cloud_rate

    def get_df_ratio(self, processes=1):
        if processes == 1:
            list_out = []
            for i, patch_name in tqdm(enumerate(self.list_patch_names)):
                list_out.append(self.get_image_type(patch_name=patch_name))
                # patch_name, ratio, cloud_rate = self.get_image_type(patch_name=patch_name)
                # list_out.append([patch_name, ratio, cloud_rate])
            df = pd.DataFrame(list_out, columns=['patch_name', 'non_null_rate', 'cloud_rate'])
        elif processes > 1:
            func_args = [(self.get_image_type, patch_name) for patch_name in self.list_patch_names]
            list_out = imap_unordered_bar(argwrapper, func_args, n_processes=processes)
            df = pd.DataFrame(list_out, columns=['patch_name', 'non_null_rate', 'cloud_rate'])
            df_temp = pd.DataFrame(np.array(self.list_patch_names).T, columns=['patch_name'])
            df = pd.merge(df_temp, df, on='patch_name', how='left')

        else:
            df = None

        return df


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

    non_null_rate = 1.0
    cloud_rate = None
    processes = 12

    # set transform
    transforms_train = get_training_augmentation()  # albu.Compose([albu.HorizontalFlip()])
    transforms_test = None  # get_validation_augmentation()

    # preprocessing if smp
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn=preprocessing_fn)
    if include_nir:
        preprocessing = None

    dataset = L8CLoudDataset(base_dir=base_dir, datatype=datatype, transforms=transforms_train,
                             preprocessing=preprocessing, include_nir=include_nir,
                             non_null_rate=non_null_rate,
                             cloud_rate=cloud_rate,
                             processes=processes)
    img, mask = dataset.__getitem__(100)
    # dataset.get_image_type(100)
    print(img.shape, mask.shape)
    #
    # df = dataset.get_df_ratio(processes=10)

    # divide training set and validation set
    n_samples = len(dataset)  # n_samples is 60000
    train_size = int(len(dataset) * train_ratio)  # train_size is 48000
    val_size = n_samples - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(len(train_ds))
    print(len(valid_ds))


if __name__ == "__main__":
    main()
