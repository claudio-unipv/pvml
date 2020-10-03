#!/usr/bin/env python3

import torch
import torchvision
import torchvision.datasets
import pvml
import numpy as np
import argparse
import os


# Training pvmlnet with pvml would take too much time.  This is why
# pytorch is used instead.  This way it is possible to use a GPU.  At
# the end the learned weights are copied in the pvml CNN.
#
# Suggested training schedule:
# - 30 epocs with learning rate 0.01
# - 30 epocs with learning rate 0.001
# - 30 epocs with learning rate 0.0001
#
# Accuracy for ilsvrc2012:
# - top1: 56.471
# - top5: 80.289
#


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_pvmlnet():
    """Create the network as a pytorch model."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 96, 7, 4, padding=3),
        torch.nn.BatchNorm2d(96),
        torch.nn.ReLU(),
        torch.nn.Conv2d(96, 192, 3, 2, padding=1),
        torch.nn.BatchNorm2d(192),
        torch.nn.ReLU(),
        torch.nn.Conv2d(192, 192, 3, 1, padding=1),
        torch.nn.BatchNorm2d(192),
        torch.nn.ReLU(),
        torch.nn.Conv2d(192, 384, 3, 2, padding=1),
        torch.nn.BatchNorm2d(384),
        torch.nn.ReLU(),
        torch.nn.Conv2d(384, 384, 3, 1, padding=1),
        torch.nn.BatchNorm2d(384),
        torch.nn.ReLU(),
        torch.nn.Conv2d(384, 512, 3, 2, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, 3, 1, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(),

        torch.nn.Conv2d(512, 512, 3, 2, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, 3, 1, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(),
        
        torch.nn.Conv2d(512, 1024, 4, 1, padding=0),
        torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(),
        torch.nn.Conv2d(1024, 1024, 1, 1, padding=0),
        torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(),
        torch.nn.Conv2d(1024, 1000, 1, 1, padding=0),
        torch.nn.AdaptiveAvgPool2d(1)
    )
    return model
    

def ___make_pvmlnet():
    """Mirror pvmlnet as a pytorch model."""
    layers = [torch.nn.ConstantPad2d(16, 0.0)]
    in_channels = 3
    for channels, size, stride in LAYERS:
        conv = torch.nn.Conv2d(in_channels, channels, size, stride, padding=0)
        layers.append(conv)
        layers.append(torch.nn.ReLU())
        in_channels = channels
    layers[-1] = torch.nn.AdaptiveAvgPool2d(1)
    layers[-6:-6] = [torch.nn.Dropout2d()]
    layers[-4:-4] = [torch.nn.Dropout2d()]
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
    a('-b', '--batch-size', default=256, type=int, metavar='N',
      help='mini-batch size (default: 256)')
    a('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
      help='initial learning rate', dest='lr')
    a('--momentum', default=0.9, type=float, metavar='M',
      help='momentum')
    a('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
      help='weight decay (default: 1e-4)', dest='weight_decay')
    a("--start-from", help="Start from a pretrained model")
    a('-p', '--print-freq', default=100, type=int, metavar='N',
      help='print frequency (default: 100)')
    a('--seed', default=None, type=int,
      help='seed for initializing training. ')
    a('--gpu', default=0, type=int, help='GPU id to use.')
    return parser.parse_args()


def export(ptfile, npfile):
    # net = make_pvmlnet()
    net = torch.load(ptfile, map_location=torch.device('cpu'))
    net2 = pvml.PVMLNet()
    print(net)
    nw = 0
    nb = 0
    for k, v in net.named_parameters():
        if "weight" in k:
            vv = np.transpose(v.detach().numpy(), (2, 3, 1, 0))
            net2.weights[nw][...] = vv
            nw += 1
        elif "bias" in k:
            vv = v.detach().numpy()
            net2.biases[nb][...] = vv
            nb += 1
    if nw != len(net2.weights) or nb != len(net2.biases):
        raise RuntimeError("Wrong number of parameters")
    net2.save(npfile)
    

def main():
    args = parse_args()

    #----------------------------------------------------------------------
    # Setup data
    #----------------------------------------------------------------------

    tr = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])

    train_dataset = torchvision.datasets.ImageNet(args.data, split='train',
                                                  transform=tr)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    tr = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = torchvision.datasets.ImageNet(args.data, split='val',
                                                transform=tr)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #----------------------------------------------------------------------
    # Setup models and optimizer
    #----------------------------------------------------------------------

    if args.start_from is None:
        model = make_pvmlnet()
    else:
        model = torch.load(args.start_from)
    model = model.cuda(args.gpu)
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #----------------------------------------------------------------------
    # Training loop
    #----------------------------------------------------------------------

    best_acc1 = 0
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        torch.save(model, "pvmlnet_last.pt")
        save_pvmlnet(model, "pvmlnet_last.npz")
        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model, "pvmlnet_best.pt")
            save_pvmlnet(model, "pvmlnet_best.npz")


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        output = model(images).squeeze(-1).squeeze(-1)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print("[{}] {}: {:.3f} {:.3f} {:.3f}".format(epoch, i, loss.item(), acc1.item(), acc5.item()))


def validate(val_loader, model, criterion, args):
    model.eval()
    tot1 = 0
    tot5 = 0
    count = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output = model(images).squeeze(-1).squeeze(-1)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            tot1 += acc1.item()
            tot5 += acc5.item()
            count += 1
        print(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(tot1 / count, tot5 / count))
    return tot1 / count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def embed_normalization(net):
    """Include normalization in the model."""
    # c * ((x - m) / s) + b   becomes   c' * x + b'
    #   with   c' = c / s   and   b' = b - c * (m / s)
    x = torch.randn(1, 3, 224, 224)
    mean = torch.tensor(MEAN)
    std = torch.tensor(STD)
    c = net[0]
    y = (mean / std).view(1, 3, 1, 1).repeat(1, 1, c.weight.size(2), c.weight.size(3))
    bb = torch.nn.functional.conv2d(y, c.weight.data, padding=0)
    breakpoint()
    c.weight.data /= std.view(1, 3, 1, 1)
    c.bias.data -= bb.data[0, :, 0, 0]


# the fuse_convbn function code is refered from https://zhuanlan.zhihu.com/p/49329030
def fuse_convbn(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 bias=True)
    fused_conv.weight = torch.nn.Parameter(w)
    fused_conv.bias = torch.nn.Parameter(b)
    return fused_conv


def fuse_modules(net):
    to_fuse = []
    mods = []
    for i in range(len(net) - 1):
        if isinstance(net[i], torch.nn.Conv2d) and isinstance(net[i + 1], torch.nn.BatchNorm2d):
            mods.append(fuse_convbn(net[i], net[i + 1]))
        elif not isinstance(net[i], torch.nn.BatchNorm2d) and not isinstance(net[i], torch.nn.Dropout2d):
            mods.append(net[i])
    return torch.nn.Sequential(*mods)


def save_pvmlnet(net, filename):
    pnet = convert_model(net)
    pnet.save(filename)

    
def convert_model(net):
    """Pytorch to pvml conversion."""
    net = fuse_modules(net).to("cpu")
    embed_normalization(net)
    net.eval()
    channels = [3] + [c.out_channels for c in net if isinstance(c, torch.nn.Conv2d)]
    kernel_sz = [c.weight.size(2) for c in net if isinstance(c, torch.nn.Conv2d)]
    strides = [c.stride[0] for c in net if isinstance(c, torch.nn.Conv2d)]
    pads = [c.padding[0] for c in net if isinstance(c, torch.nn.Conv2d)]
    pnet = pvml.CNN(channels, kernel_sz, strides, pads)
    k = 0
    for i in range(len(net)):
        if isinstance(net[i], torch.nn.Conv2d):
            v = net[i].weight
            pnet.weights[k][...] = np.transpose(v.detach().numpy(), (2, 3, 1, 0))
            b = net[i].bias
            pnet.biases[k][...] = b.detach().numpy()
            k += 1
    return pnet

    
def print_network(net):
    """Print the size of activations after each convolutional layer."""
    x = torch.zeros((1, 3, 224, 224))
    print("x".join(map(str, x.size())))
    handles = []
    for m in net.children():
        if isinstance(m, torch.nn.Conv2d):
            h = m.register_forward_hook(lambda m, i, o: print("x".join(map(str, o.size()))))
            handles.append(h)
    train = net.training
    net.eval()
    net(x)
    if train:
        net.train()
    for h in handles:
        h.remove()


if __name__ == "__main__":
    net = make_pvmlnet()
    params = sum(p.numel() for p in net.parameters())
    print("{:.1f}M parameters".format(params / 1000000))
    print()
    print_network(net)
    print()
    main()
    # # export("pvmlnet_90.pt", "pvmlnet.npz")
    # net = make_pvmlnet()
    # print(net)
    # x = torch.rand(4, 3, 224, 224)
    # y = net(x)
    # print(x.size(), "->", y.size())
    # net2 = pvml.make_pvmlnet()
    # x = np.zeros((4, 224, 224, 3))
    # a = net2.forward(x)
    # for aa in a:
    #     print("x".join(map(str, aa.shape)))
