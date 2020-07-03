import os
from tqdm import tqdm, trange
import shutil
from collections import namedtuple
import gc
import time
import numpy as np
import cv2
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import models, transforms
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import segmentation_models_pytorch as smp

from catalyst.dl.runner import SupervisedRunner

from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl import utils
from datasets.dataset_38_cloud import get_preprocessing, get_training_augmentation, get_validation_augmentation, \
    L8CLoudDataset

def train(n_epochs, **train_kwargs):
    print()

from models.unet_test import UNET

def main():
    base_dir = '/dataset/kaggle/38-cloud'
    datatype = 'train'  # 'test'
    include_nir = True
    train_ratio = 0.8
    batch_size = 12
    num_workers = 0
    lr = 0.01
    debug = False
    resume = None

    # model
    # todo: resolve or TorchSat method
    model_cls = UNET
    model_kwargs = {
        'in_channels': 4,
        'out_channels': 2
    }
    epochs = 10

    # todo: add class name
    log_base_dir = "./logs/38_cloud"

    tb_log_dir = os.path.join(log_base_dir, "tb")
    dir_ckp = os.path.join(log_base_dir, "checkpoints")

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

    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loaders = {
        "train": train_dl,
        "valid": valid_dl
    }

    # check
    xb, yb = next(iter(train_dl))
    print(xb.shape, yb.shape)

    if debug:
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model, criterion, optimizer
    model = model_cls(**model_kwargs)
    pred = model(xb)
    print(pred.shape)

    model.to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device))

    print(model)
    print(summary(model, input_size=tuple(xb.shape[1:])))

    # optimizer = torch.optim.Adam([
    #     {'params': model.decoder.parameters(), 'lr': 1e-2},
    #     {'params': model.encoder.parameters(), 'lr': 1e-3},
    # ])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Tensorboard
    writer = SummaryWriter(log_dir=tb_log_dir)
    # display some examples in tensorboard
    images, labels = next(iter(train_dl))
    # originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    writer.add_images('images/original', images[:,:3,...], 0)
    # writer.add_images('images/normalized', images, 0)
    writer.add_graph(model, images.to(device))

    # todo: if pretrained

    # train
    # normal pytorch procedure
    # todo: check Satorch and save results
    train_loss, valid_loss = train_org(model, train_dl, valid_dl, criterion, optimizer, device, acc_metric, dir_ckp,
                                       scheduler=None, epochs=epochs, writer=writer)
    # todo: check tensorboard
    # todo: Catalyst
    # todo: Catalyst with smp
    print(train_loss)
    print(valid_loss)



def train_org(model, train_dl, valid_dl, criterion, optimizer, device, acc_metric, dir_ckp, scheduler=None,
              epochs=50, writer=None):
    start = time.time()
    train_loss, valid_loss = [], []
    best_acc = 0.0

    for epoch in trange(epochs, desc="Epochs"):
        metrics_train = train_epoch(model, train_dl, criterion, optimizer, device, acc_metric, epoch, grad_acc=1, phase='train', writer=writer)
        # todo: check
        # writer = metrics_train['writer']
        metrics_valid = train_epoch(model, valid_dl, criterion, optimizer, device, acc_metric, epoch, grad_acc=1, phase='valid', writer=writer)
        # writer = metrics_train['writer']

        train_loss.append(metrics_train['loss'])
        valid_loss.append(metrics_valid['loss'])

        if scheduler is not None:
            writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
            scheduler.step(metrics_valid['loss'])
        print(f'epoch: {epoch} ', metrics_train['loss'], metrics_valid['loss'])

        torch.save(model.state_dict(), os.path.join(dir_ckp, 'cls_epoch_{}.pth'.format(epoch)))

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def train_epoch(model, data_loader, criterion, optimizer, device, acc_fn, epoch, grad_acc=1, phase='train', writer=None):
    if phase == 'train':
        model.train()
        # zero the parameter gradients
        optimizer.zero_grad()

    running_loss = 0.0
    running_acc = 0.0

    total_loss = 0.

    # if show_progress:
    #     data_loader = tqdm(data_loader, phase, unit="batch")
    # for i, (inputs, labels) in enumerate(data_loader):
    for i, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if phase == 'train':
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient accumulation
            if (i % grad_acc) == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        else:
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                total_loss += loss.item()

        acc = acc_fn(outputs, labels)
        running_acc += acc * data_loader.batch_size

        if writer is not None:
            writer.add_scalar('phase/loss', loss.item(), len(data_loader) * epoch + i)

    epoch_loss = total_loss / len(data_loader.dataset)
    epoch_acc = running_acc / len(data_loader.dataset)
    metrics = {'loss': epoch_loss, 'acc': epoch_acc}

    if writer is not None:
        metrics['writer'] = writer

    return metrics


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

if __name__ == "__main__":
    main()