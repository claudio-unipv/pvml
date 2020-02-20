#!/usr/bin/env python3

import torch
import torchvision
import torchvision.datasets
import pvml
from pvml.pvmlnet import _LAYERS as LAYERS
import numpy as np
import argparse
import os


# Training pvmlnet with pvml would take too much time.  This is why
# pytorch is used instead.  This way it is possible to access to use a
# GPU.  At the end the learned weights are copied in the pvml CNN.


def make_pvmlnet():
    layers = []
    in_channels = 3
    for channels, size, stride in LAYERS:
        conv = torch.nn.Conv2d(in_channels, channels, size, stride)
        layers.append(conv)
        layers.append(torch.nn.ReLU())
        in_channels = channels
    layers[-1] = torch.nn.AdaptiveAvgPool2d(1)
    layers[-4:-4] = [torch.nn.Dropout2d()]
    layers[-2:-2] = [torch.nn.Dropout2d()]
    torch.nn.init.constant_(layers[0].bias, -0.5)
    model = torch.nn.Sequential(*layers)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='PVMLNET Training')
    a = parser.add_argument
    a('data', metavar='DIR', help='path to dataset')
    a('-j', '--workers', default=4, type=int, metavar='N',
      help='number of data loading workers (default: 4)')
    a('--epochs', default=90, type=int, metavar='N',
      help='number of total epochs to run')
    a('-b', '--batch-size', default=256, type=int,
      metavar='N',
      help='mini-batch size (default: 256)')
    a('--lr', '--learning-rate', default=0.1, type=float,
      metavar='LR', help='initial learning rate', dest='lr')
    a('--momentum', default=0.9, type=float, metavar='M',
      help='momentum')
    a('--wd', '--weight-decay', default=1e-4, type=float,
      metavar='W', help='weight decay (default: 1e-4)',
      dest='weight_decay')
    a('-p', '--print-freq', default=10, type=int,
      metavar='N', help='print frequency (default: 10)')
    a('--seed', default=None, type=int,
      help='seed for initializing training. ')
    a('--gpu', default=None, type=int,
      help='GPU id to use.')
    return parser.parse_args()


def train():
    args = parse_args()
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # train_dataset = torchvision.datasets.ImageFolder(
    #     traindir,

    tr = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])

    # !!! train_dataset = torchvision.datasets.ImageNet(args.data, split='train',
                                                  transform=tr)
    train_dataset = torchvision.datasets.ImageNet(args.data, split='val',
                                                  transform=tr)

    print(type(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    net_pt = make_pvmlnet()
    net_pvml = pvml.make_pvmlnet(pretrained=False)



    
if __name__ == "__main__":
    train()
    # net = make_pvmlnet()
    # x = torch.zeros(4, 3, 224, 224)
    # y = net(x)
    # print(x.size(), "->", y.size())
    # net2 = pvml.pvmlnet()
    # x = np.zeros((4, 224, 224, 3))
    # a = net2.forward(x)
    # for aa in a:
    #     print("x".join(map(str, aa.shape)))
