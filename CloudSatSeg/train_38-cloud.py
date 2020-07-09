import os

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
from calc.trainer import acc_metric, train_org


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
    resume = None

    # log
    log_base_dir = "./logs/38_cloud_test"

    non_null_rate = 1.0
    cloud_rate = None
    processes = 12

    torch.backends.cudnn.benchmark = True
    if debug:
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # transform
    transforms_train = get_transform(name=name_trans_train, **kwargs_trans)

    # preprocessing
    preprocessing = None
    # ENCODER = 'resnet50'
    # ENCODER_WEIGHTS = 'imagenet'
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # preprocessing = get_preprocessing(preprocessing_fn=preprocessing_fn)

    # dataset
    dataset = L8CLoudDataset(base_dir=base_dir, datatype=datatype_train, transforms=transforms_train,
                             preprocessing=preprocessing, include_nir=include_nir,
                             non_null_rate=non_null_rate,
                             cloud_rate=cloud_rate,
                             processes=processes)
    # divide training set and validation set
    n_samples = len(dataset)  # n_samples is 60000
    train_size = int(len(dataset) * train_ratio)  # train_size is 48000
    val_size = n_samples - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loaders = {
        "train": train_dl,
        "valid": valid_dl
    }

    # model
    model = get_model(name=model_name, out_channels=out_channels, **kwargs_model)
    model.to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device))

    # check model
    xb, yb = next(iter(train_dl))
    print(xb.shape, yb.shape)
    print(model)
    print(summary(model, input_size=tuple(xb.shape[1:])))

    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # optim and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)

    # log
    tb_log_dir = os.path.join(log_base_dir, "tb")
    dir_ckp = os.path.join(log_base_dir, "checkpoints")
    if not os.path.exists(dir_ckp):
        os.makedirs(dir_ckp)
    # Tensorboard
    writer = SummaryWriter(log_dir=tb_log_dir)
    # display some examples in tensorboard
    images, labels = next(iter(train_dl))
    # originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    writer.add_images('images/original', images[:, :3, ...], 0)
    # writer.add_images('images/normalized', images, 0)
    writer.add_graph(model, images.to(device))

    # train
    # normal pytorch procedure
    # todo: check Satorch and save results
    train_loss, valid_loss = train_org(model, train_dl, valid_dl, criterion, optimizer, device, acc_metric, dir_ckp,
                                       scheduler=lr_scheduler, epochs=epochs, writer=writer)

    # # pred todo: this should be moved to other code
    # # dataset
    # dataset = L8CLoudDataset(base_dir=base_dir, datatype=datatype_test, transforms=transforms_train,
    #                          preprocessing=preprocessing, include_nir=include_nir,
    #                          non_null_rate=non_null_rate,
    #                          cloud_rate=cloud_rate,
    #                          processes=processes)

    # todo: Catalyst
    # todo: Catalyst with smp
    # todo: ignite ???
    print(train_loss)
    print(valid_loss)

    # todo: pred
    # todo: check accuracy measure

if __name__ == "__main__":
    main()
