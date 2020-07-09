import os
from tqdm import tqdm, trange
import time

import torch

# todo: https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/pytorch/UNet_Seg_VOC2012_pytorch.ipynb check

def test_org(model, loader, device, criterion, acc_fn):
    model.eval()
    model.to(device)

    running_acc = 0.0

    total_loss = 0.

    for i, (inputs, labels) in tqdm(enumerate(loader), total=len(loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
        acc = acc_fn(outputs, labels)
        running_acc += acc * loader.batch_size

    total_loss = total_loss / len(loader.dataset)
    total_acc = running_acc / len(loader.dataset)
    # metrics = {'loss': total_loss, 'acc': total_acc}

    return total_loss


def train_org(model, train_dl, valid_dl, criterion, optimizer, device, acc_metric, dir_ckp, scheduler=None,
              epochs=50, writer=None):
    start = time.time()
    train_loss, valid_loss = [], []
    best_acc = 0.0

    for epoch in trange(epochs, desc="Epochs"):
        metrics_train = train_epoch(model, train_dl, criterion, optimizer, device, acc_metric, epoch, grad_acc=1,
                                    phase='train', writer=writer)
        # todo: check
        # writer = metrics_train['writer']
        metrics_valid = train_epoch(model, valid_dl, criterion, optimizer, device, acc_metric, epoch, grad_acc=1,
                                    phase='valid', writer=writer)
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


def train_epoch(model, data_loader, criterion, optimizer, device, acc_fn, epoch, grad_acc=1, phase='train',
                writer=None):
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
