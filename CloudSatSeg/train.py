import os
from tqdm import tqdm, trange
import shutil
from collections import namedtuple
import gc


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

from datasets.kaggle_cloud_org import get_meta_info_table, prepare_dataset, CloudDataset, get_preprocessing, \
    get_training_augmentation, get_validation_augmentation
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl import utils


def dice_no_threshold(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 1e-7,
        threshold: float = None,
        # activation: str = "Sigmoid"
):
    # activation_fn = get_activation_fn(activation)
    # outputs = activation_fn(outputs)
    outputs = torch.sigmoid(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice


def train_epoch(model, data_loader, criterion, optimizer, device, grad_acc=1):
    model.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    total_loss = 0.
    for i, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient accumulation
        if (i % grad_acc) == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    total_loss /= len(data_loader)
    metrics = {'train_loss': total_loss}
    return metrics


def eval_epoch(model, data_loader, criterion, device):
    model.eval()
    num_correct = 0.
    valid_loss = 0.0
    dice_score = 0.0

    with torch.no_grad():
        total_loss = 0.
        for inputs, labels in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # todo: torchmax de iino?
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            dice_cof = dice_no_threshold(outputs.cpu(), labels.cpu()).item()

            valid_loss += loss.item()
            dice_score += dice_cof
            # num_correct += torch.sum(preds == labels.data)

        valid_loss /= len(data_loader.dataset)
        # num_correct /= len(data_loader.dataset)
        dice_score /= len(data_loader.dataset)
        metrics = {'valid_loss': valid_loss, 'dice_score': dice_score}
    return metrics


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def main_kaggle_smp(path_dataset='/dataset/kaggle/understanding_cloud_organization',
                    ENCODER='resnet50',
                    ENCODER_WEIGHTS='imagenet',
                    num_workers=0,
                    batch_size=8,
                    epochs=19,
                    debug=False,
                    exec_catalyst=True,
                    logdir="/src/logs/segmentation",
                    pretrained=True
                    ):
    # below line is potential input args
    # (name_dataset='eurosat', lr=0.0001, wd=0, ratio=0.9, batch_size=32, workers=4, epochs=15, num_gpus=1,
    # resume=None, dir_weights='./weights'):

    torch.backends.cudnn.benchmark = True

    # Dataset
    train, sub = get_meta_info_table(path_dataset)
    train_ids, valid_ids, test_ids = prepare_dataset(train, sub)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn), path=path_dataset)
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids,
                                 transforms=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn), path=path_dataset)
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    # todo: check how to used device in this case
    DEVICE = 'cuda'
    if debug:
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ACTIVATION = None
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    images, labels = next(iter(train_loader))
    model.to(device)
    print(model)
    print(summary(model, input_size=tuple(images.shape[1:])))

    # use smp epoch
    # num_epochs = 19

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.DiceLoss(eps=1.)  # smp.utils.losses.BCEDiceLoss(eps=1.)

    if not pretrained:
        # catalyst
        if exec_catalyst:
            device = utils.get_device()
            runner = SupervisedRunner(device=device)

            # train model
            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                loaders=loaders,
                callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
                logdir=logdir,
                num_epochs=epochs,
                verbose=True
            )

            # # prediction
            # encoded_pixels = []
            # loaders = {"infer": valid_loader}
            # runner.infer(
            #     model=model,
            #     loaders=loaders,
            #     callbacks=[
            #         CheckpointCallback(
            #             resume=f"{logdir}/checkpoints/best.pth"),
            #         InferCallback()
            #     ],
            # )
            # valid_masks = []
            #
            # # todo: where .pth?
            # # todo: from here
            # valid_num = valid_dataset.__len__()
            # probabilities = np.zeros((valid_num * 4, 350, 525))
            # for i, (batch, output) in enumerate(tqdm(zip(
            #         valid_dataset, runner.callbacks[0].predictions["logits"]))):
            #     image, mask = batch
            #     for m in mask:
            #         if m.shape != (350, 525):
            #             m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            #         valid_masks.append(m)
            #
            #     for j, probability in enumerate(output):
            #         if probability.shape != (350, 525):
            #             probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            #         probabilities[valid_num * 4 + j, :, :] = probability
            #
            # # todo: from here
            # class_params = {}
            # for class_id in range(4):
            #     print(class_id)
            #     attempts = []
            #     for t in range(0, 100, 5):
            #         t /= 100
            #         for ms in [0, 100, 1200, 5000, 10000]:
            #             masks = []
            #             for i in range(class_id, len(probabilities), 4):
            #                 probability = probabilities[i]
            #                 predict, num_predict = post_process(sigmoid(probability), t, ms)
            #                 masks.append(predict)
            #
            #             d = []
            #             for i, j in zip(masks, valid_masks[class_id::4]):
            #                 if (i.sum() == 0) & (j.sum() == 0):
            #                     d.append(1)
            #                 else:
            #                     d.append(dice(i, j))
            #
            #             attempts.append((t, ms, np.mean(d)))
            #
            #     attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
            #
            #     attempts_df = attempts_df.sort_values('dice', ascending=False)
            #     print(attempts_df.head())
            #     best_threshold = attempts_df['threshold'].values[0]
            #     best_size = attempts_df['size'].values[0]
            #
            #     class_params[class_id] = (best_threshold, best_size)

        else:
            for epoch in trange(epochs, desc="Epochs"):
                metrics_train = train_epoch(model, train_loader, criterion, optimizer, device)
                metrics_eval = eval_epoch(model, valid_loader, criterion, device)

                scheduler.step(metrics_eval['valid_loss'])
                print(f'epoch: {epoch} ', metrics_train, metrics_eval)
    else:
        if exec_catalyst:
            device = utils.get_device()
            checkpoint = utils.load_checkpoint(f'{logdir}/checkpoints/best_full.pth')
            utils.unpack_checkpoint(checkpoint, model=model)
            runner = SupervisedRunner(model=model)

            # prediction with infer
            encoded_pixels = []
            loaders = {"infer": valid_loader}
            runner.infer(
                model=model,
                loaders=loaders,
                callbacks=[
                    CheckpointCallback(
                        resume=f"{logdir}/checkpoints/best.pth"),
                    InferCallback()
                ],
            )

            # todo: jupyterで確認中
            valid_masks = []


            valid_num = valid_dataset.__len__()
            probabilities = np.zeros((valid_num * 4, 350, 525))
            for i, (batch, output) in enumerate(tqdm(zip(
                    valid_dataset, runner.callbacks[0].predictions["logits"]))):
                image, mask = batch
                for m in mask:
                    if m.shape != (350, 525):
                        m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                    valid_masks.append(m)

                for j, probability in enumerate(output):
                    if probability.shape != (350, 525):
                        probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                    probabilities[i * 4 + j, :, :] = probability

            class_params = {}
            for class_id in range(4):
                print(class_id)
                attempts = []
                for t in range(0, 100, 5):
                    t /= 100
                    for ms in [0, 100, 1200, 5000, 10000]:
                        masks = []
                        for i in range(class_id, len(probabilities), 4):
                            probability = probabilities[i]
                            predict, num_predict = post_process(sigmoid(probability), t, ms)
                            masks.append(predict)

                        d = []
                        for i, j in zip(masks, valid_masks[class_id::4]):
                            if (i.sum() == 0) & (j.sum() == 0):
                                d.append(1)
                            else:
                                d.append(dice(i, j))

                        attempts.append((t, ms, np.mean(d)))

                attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

                attempts_df = attempts_df.sort_values('dice', ascending=False)
                print(attempts_df.head())
                best_threshold = attempts_df['threshold'].values[0]
                best_size = attempts_df['size'].values[0]

                class_params[class_id] = (best_threshold, best_size)

            # predictions
            torch.cuda.empty_cache()
            gc.collect()

            test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids, transforms=get_validation_augmentation(),
                                        preprocessing=get_preprocessing(preprocessing_fn))
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

            loaders = {"test": test_loader}
            encoded_pixels = []
            image_id = 0
            for i, test_batch in enumerate(tqdm(loaders['test'])):
                runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
                for i, batch in enumerate(runner_out):
                    for probability in batch:

                        probability = probability.cpu().detach().numpy()
                        if probability.shape != (350, 525):
                            probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                        predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
                                                            class_params[image_id % 4][1])
                        if num_predict == 0:
                            encoded_pixels.append('')
                        else:
                            r = mask2rle(predict)
                            encoded_pixels.append(r)
                        image_id += 1

            sub['EncodedPixels'] = encoded_pixels
            sub.to_csv('data/kaggle_cloud_org/submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)



if __name__ == "__main__":
    dataset = 'kaggle'
    pre_trained = True

    if dataset == 'kaggle':
        main_kaggle_smp(exec_catalyst=True, pretrained=pre_trained)
