import argparse
import os
import random
import time
import copy

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

# import cutom helpers
from utils import images_to_probs, plot_classes_preds, matplotlib_imshow, plot_kernels_tensorboard
from featuremaps import extract_feature_map
# import models
from data import get_data

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# was getting weird errors on my machine as well
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# only sampling some kernels per layer bc there are tens of thousands of kernels in some layers
num_kernels_to_sample = 16

def create_parser():
    # training configurations
    parser = argparse.ArgumentParser(description='classifier args')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size')
    parser.add_argument('--niter', type=int, default=2,
                        help='number of epochs to train for')
    parser.add_argument('--seed', default=0, type=int,
                        help='manual seed')
    parser.add_argument('--checkpoint', type=str,
                        help='continue training from a checkpoint')
    parser.add_argument('--dataname', type=str,
                        default='CIFAR10')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for SGD')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--train_model', type=str, default='n',
                        help='y to train model, n to use pretrained')
    parser.add_argument('--name', type=str, default='1',
                        help='name to identify run')
    return parser

def main(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    trainset = get_data(opt.dataname)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=4)

    # net = models.Net()
    if opt.train_model == 'y':
        use_pretrained = False
    else:
        use_pretrained = True

    net = models.resnet50(pretrained=use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
    writer = SummaryWriter(f'runs/{opt.dataname}_experiment_{opt.name}')

    all_layers = []

    # Filter out all the layers with no weights for kernel viz
    for module in list(net.modules()):
        module_type = type(module)
        if module_type not in {nn.Sequential, models.resnet.ResNet, nn.ReLU, nn.BatchNorm2d,
                               models.resnet.Bottleneck, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                               nn.modules.pooling.AvgPool2d}:
            all_layers.append(module)
            # print(module.weight.shape)

    log_interval = 50
    # running_loss = 0.0
    if not use_pretrained:
        for epoch in range(opt.niter):  # loop over the dataset multiple times if training
            print(f'Starting epoch: {epoch + 1}')
            for i, data in enumerate(tqdm(trainloader, 0)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                # print(loss.item())
                # loss = Variable(loss, requires_grad = True)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()

                if i % log_interval == 10:
                    # log the running loss
                    writer.add_scalar('training loss',
                                    loss.item(),
                                    epoch * len(trainloader) + i)

                    # log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch
                    # writer.add_figure('predictions vs. actuals',
                    #                 plot_classes_preds(net, inputs, labels),
                    #                 global_step=epoch * len(trainloader) + i)
                    # running_loss = 0.0

                    # Log non 1x1 kernels to tensorboard

                    j = 0
                    print("Writing kernel for epoch %d step %d" % (epoch, i))
                    for layer in all_layers[:-1]: 
                        layer2 = layer.weight.data.cpu().numpy()
                        if layer2.shape[2] > 1:
                            fig = plot_kernels_tensorboard(layer2, num_kernels_to_sample)
                            writer.add_figure("Kernels for layer %d" % (j),
                                        fig,
                                        epoch * len(trainloader) + i)
                        j += 1

                    # extract feature map
                    print("Writing feature map for epoch %d step %d" % (epoch, i))
                    for layer_id in range(4):
                        writer.add_figure(f'Feature map for layer {layer_id}',
                                            extract_feature_map(net, layer_id, device),
                                            epoch * len(trainloader) + i)
    else: # write figures to tensorboard
        j = 0
        for layer in all_layers[:-1]:
            layer2 = layer.weight.clone().detach().numpy()
            if layer2.shape[2] > 1:
                fig = plot_kernels_tensorboard(layer2, num_kernels_to_sample)
                writer.add_figure("Kernels for layer %d" % (j),
                            fig)
            j += 1

        # extract feature maps
        for layer_id in range(4):
            fig = extract_feature_map(net, layer_id)
            writer.add_figure(f'Feature map for layer {layer_id}',
                                fig)
    time.sleep(3)

    print(f'Finished Training')

if __name__ == '__main__':
    opt = create_parser().parse_args()
    print(opt)
    main(opt)
