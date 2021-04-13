import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import images_to_probs, plot_classes_preds, matplotlib_imshow
import models
from data import get_data

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


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
                        default='fashion')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for SGD')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    return parser

def main(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    trainset = get_data(opt.dataname)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=0)

    net = models.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
    writer = SummaryWriter(f'runs/{opt.dataname}_experiment_1')

    running_loss = 0.0
    for epoch in range(opt.niter):  # loop over the dataset multiple times
        print(f'Starting epoch: {epoch + 1}')
        for i, data in enumerate(tqdm(trainloader, 0)):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                # log the running loss
                writer.add_scalar('training loss',
                                running_loss / 1000,
                                epoch * len(trainloader) + i)

                # log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                plot_classes_preds(net, inputs, labels),
                                global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
    print(f'Finished Training, loss = {running_loss / 1000}')

if __name__ == '__main__':
    opt = create_parser().parse_args()
    print(opt)
    main(opt)