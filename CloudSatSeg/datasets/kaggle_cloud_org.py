'''
Ref: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools
'''
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

train_on_gpu = True


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def get_meta_info_table(path, plot=True):
    print(os.listdir(path))
    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')
    print(train.head())
    n_train = len(os.listdir(f'{path}/train_images'))
    n_test = len(os.listdir(f'{path}/test_images'))
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')

    print(train['Image_Label'].apply(lambda x: x.split('_')[1]).value_counts())
    print(train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[1]).value_counts())
    print(train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts().value_counts())
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

    if plot:
        plot_train(train, path=path)

    return train, sub


def plot_train(train, path, path_out='data/kaggle_cloud_org/temp_figure/train_random.png'):
    fig = plt.figure(figsize=(25, 16))
    for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):
        for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):
            ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])
            im = Image.open(f"{path}/train_images/{row['Image_Label'].split('_')[0]}")
            plt.imshow(im)
            mask_rle = row['EncodedPixels']
            try:  # label might not be there!
                mask = rle_decode(mask_rle)
            except:
                mask = np.zeros((1400, 2100))
            plt.imshow(mask, alpha=0.5, cmap='gray')
            ax.set_title(f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['label']}")
    dir_out = os.path.dirname(path_out)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    plt.savefig(path_out)
    return


def prepare_dataset(train, sub):
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts(). \
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42,
                                            stratify=id_mask_count['count'], test_size=0.1)
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
    return train_ids, valid_ids, test_ids


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing=None, path=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(320, 640)
    ]
    return albu.Compose(train_transform)


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
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

def main():
    path = '/dataset/kaggle/understanding_cloud_organization'

    train, sub = get_meta_info_table(path)
    train_ids, valid_ids, test_ids = prepare_dataset(train, sub)

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    num_workers = 0
    bs = 16
    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids,
                                 transforms=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))

if __name__ == "__main__":
    main()

