import os
import sys
import re
import datetime
import time
import numpy
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_training_dataloader(train_path,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], batch_size=256, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    training_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    training_loader = DataLoader(
        training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,pin_memory=True)

    return training_loader

def get_test_dataloader(test_path,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], batch_size=256, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path,transform=transform_test)
    test_loader = DataLoader(
        test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader



def train(epoch):
    checkpoint_path = os.path.join("./places_standard/", '{net}-epoch1_{iter}-{type}.pth')
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
        logging.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        loss.item(),
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        trained_samples=batch_index * args.b + len(images),
        total_samples=len(training_loader.dataset)
    ))
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))
        # print(epoch,batch_index)
        if(epoch==1 and batch_index%100==0):
            logging.info("Saving model ")
            weights_path = checkpoint_path.format(net=args.net, iter=batch_index, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))

    #add informations to tensorboard
    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet18", help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-w',type=int,default=0,help='number of workers')
    parser.add_argument('-trainpath',type=str,required=True,help='path to train dataset')
    parser.add_argument('-testpath',type=str,required=True,help='path to test dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-epoch',type=int,default=40,help='number of epochs')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()
    logging.basicConfig(filename = 'loss_big.log',filemode='w', level=logging.DEBUG)
    net = torchvision.models.__dict__["resnet18"](num_classes=365)
    net = net.cuda()
    #data preprocessing:
    epochs = args.epoch
    checkpoint_path = os.path.join("./places_standard/", '{net}-{epoch}-{type}.pth')

    train_path = args.trainpath
    test_path = args.testpath
    training_loader = get_training_dataloader(train_path,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=True
    )

    test_loader = get_test_dataloader(test_path,
        num_workers=args.w,
        batch_size=args.b
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=- 1, verbose=False)
    for epoch in range(1, epochs + 1):
        train(epoch)
        train_scheduler.step()
        acc = eval_training(epoch)
        print("Epoch %d Accuracy %f"%(epoch,acc))
        logging.info("Testing accuracy for epoch %d = %f"%(epoch,acc))
        if epoch % 2:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            # torch.save(net.state_dict(), weights_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, weights_path)