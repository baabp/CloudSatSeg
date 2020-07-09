import os
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchsummary import summary

import segmentation_models_pytorch as smp

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl import utils

from datasets.dataset_38_cloud import get_preprocessing, get_training_augmentation, get_validation_augmentation, \
    L8CLoudDataset
from transforms.utils import get_transform
from models.utils import get_model
from calc.trainer import acc_metric, test_org
from graph.plot_results import results_show


def main():
    # general
    debug = False
    epochs = 10
    batch_size = 12
    num_workers = 0
    lr = 0.01

    # dataset
    base_dir = '/dataset/kaggle/38-cloud'
    datatype_train = 'train'  # 'test'
    datatype_test = 'test'
    include_nir = True
    train_ratio = 0.8
    test_mask = False

    # transforms
    name_trans_train = 'albu_train_0'
    name_trans_val = 'albu_val_0'
    kwargs_trans = {
        'resize': None  # (384, 384)
    }
    name_preprocessing = 'xxxx'

    # model
    model_name = 'unet_0'
    out_channels = 2
    kwargs_model = {
        'in_channels': 4
    }
    resume = os.path.join("./logs/38_cloud_test/checkpoints", "cls_epoch_9.pth")

    # log
    log_base_dir = os.path.join("./logs/38_cloud_test", model_name)

    non_null_rate = 1.0
    cloud_rate = None
    processes = 1

    torch.backends.cudnn.benchmark = True
    if debug:
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # transform
    transforms_test = get_transform(name=name_trans_val, **kwargs_trans)

    # preprocessing
    preprocessing = None
    # ENCODER = 'resnet50'
    # ENCODER_WEIGHTS = 'imagenet'
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # preprocessing = get_preprocessing(preprocessing_fn=preprocessing_fn)

    # dataset
    dataset_test = L8CLoudDataset(base_dir=base_dir, datatype=datatype_test, transforms=transforms_test,
                                  preprocessing=preprocessing, include_nir=include_nir,
                                  non_null_rate=non_null_rate,
                                  cloud_rate=cloud_rate,
                                  processes=processes, test_mask=test_mask)

    # DataLoader
    test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(test_dl.__len__())

    # model
    model = get_model(name=model_name, out_channels=out_channels, **kwargs_model)
    model.to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device))

    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # check model
    xb, yb = next(iter(test_dl))
    print(xb.shape, yb.shape)
    print(model)
    print(summary(model, input_size=tuple(xb.shape[1:])))

    test_loss = test_org(model, test_dl, criterion=criterion, device=device, acc_fn=acc_metric)

    print(test_loss)
    # print figures
    dir_dest = "./temp/38_cloud_test"
    results_show(ds=dataset_test, list_index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], model=model, save=True, dir_dest=dir_dest, fname='test.png',
                 fname_time=True, show=False,
                 fig_img_size=4, cmp_input='gray',
                 cmp_out='jet', class_num=2)

    # todo: add other models from nock, old torchsat, xxxx(to many one)
    # todo: Catalyst
    # todo: Catalyst with smp
    # todo: ignite ???

    # todo: pred
    # todo: check accuracy measure


if __name__ == "__main__":
    main()
