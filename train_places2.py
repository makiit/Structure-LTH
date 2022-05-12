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
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd

from utils import *
from pruning_utils import *
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
    model.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
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
            torch.save(model.state_dict(), weights_path)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
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
    parser.add_argument('-stages',type=int,default = 3)
    parser.add_argument('-save_dir',type=str,default="./places_small_res18")

    parser.add_argument('--pruning_times', default=16, type=int, help='overall times of pruning')
    parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
    parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt,pt or pt_trans)')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
    parser.add_argument('--conv1', action="store_true", help="whether pruning&rewind conv1")
    parser.add_argument('--fc', action="store_true", help="whether rewind fc")
    parser.add_argument('--rewind_epoch', default=2, type=int, help='rewind checkpoint')
    parser.add_argument('--fillback', action="store_true")
    parser.add_argument('--fillback-slow', action="store_true")
    parser.add_argument('--criteria', default="remain", type=str, choices=['remain', 'magnitude', 'l1', 'l2', 'taylor'])

    args = parser.parse_args()
    logging.basicConfig(filename = 'loss_big.log',filemode='w', level=logging.DEBUG)
    model = torchvision.models.__dict__["resnet18"](num_classes=365)
    model = model.cuda()
    #data preprocessing:
    epochs = args.epoch
    checkpoint_path = os.path.join("./places_imp_checkpoints/", '{net}-{epoch}-{type}.pth')

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


    for state in range(0,args.stages):
        print('******************************************')
        print('pruning state', state)
        print('******************************************')

        check_sparsity(model, conv1=False)

        for epoch in range(1, 2):

            if epoch == args.rewind_epoch and args.prune_type == 'rewind_lt' and state == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{args.rewind_epoch}.pth.tar"))
                initialization = copy.deepcopy(model.state_dict())
            if epoch == args.rewind_epoch and args.prune_type == 'rewind_lt' and state == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{args.rewind_epoch}.pth.tar"))
                initialization = copy.deepcopy(model.state_dict())
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, weights_path)

            # evaluate on validation set
            # evaluate on test set

            scheduler.step()

            all_result['test'].append(acc)

            # remember best prec@1 and save checkpoint
            is_best_sa = acc  > best_sa
            best_sa = max(acc, best_sa)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initialization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
        
            # plt.plot(all_result['train'], label='train_acc')
            # plt.plot(all_result['ta'], label='val_acc')
            # plt.plot(all_result['test_ta'], label='test_acc')
            # plt.legend()
            # plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            # plt.close()

        #report result
        acc = eval_training(40) # extra forward
        check_sparsity(model, conv1=False)
        # print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))

        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        best_sa = 0
        start_epoch = 0

        pruning_model(model, args.rate, conv1=False)
        remain_weight = check_sparsity(model, conv1=False)
        current_mask = extract_mask(model.state_dict())
        for m in current_mask:
            print(current_mask[m].float().mean())

        remove_prune(model, conv1=False)

        model.load_state_dict(initialization)
        if args.fillback:
            prune_model_custom_fillback(model, current_mask, criteria=args.criteria, train_loader=train_loader)
        elif args.fillback_slow:
            prune_model_custom_fillback_slow(model, current_mask)
        else:
            prune_model_custom(model, current_mask)

        check_sparsity(model, conv1=False)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)