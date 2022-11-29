import torch
import torch.optim as optim
import sys
sys.path.append(r"D:\CS640-Project-TransUnet\TransUnet\model\networks")

from torch.optim import lr_scheduler
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable as V
import cv2
import os
import math
import warnings
from tqdm import tqdm
import numpy as np
from time import time
from shutil import copyfile, move
# from TransUnet import get_transNet
import TransUnet
from model.networks.TransUnet import get_transNet
from data import ImageFolder
from framework import MyFrame
from TransUnet.loss.dice import *
from TransUnet.loss.loss import SoftDiceLossV2
from tensorboardX import SummaryWriter

#
# import sys
# sys.path.append('../model/networks/')

model_name = 'TransUnet'
loss_name = 'dice'
writer = SummaryWriter(os.path.join('../../log/trainlog', 'segment', model_name + loss_name))


def train(transUnet, criterion, optimizer, iters):
    config_file = 'train_config.txt'
    dirs = []
    for line in open(config_file):
        dirs.append(line)

    data_root = dirs[0]
    data_root = data_root.replace('\\', '/')
    data_root = data_root.strip('\n')
    pre_model = dirs[1]
    pre_model = pre_model.replace('\\', '/')
    batch_size_ = dirs[2]
    batch_size_ = batch_size_.replace('\\', '/')
    lr = dirs[3]
    epoch_num_ = dirs[4]
    epoch_num_ = epoch_num_.replace('\\', '/')
    model_name = dirs[5]
    model_name = model_name.replace('\\', '/')
    # TODO: open the config files to get arguments

    batch_size = 1
    epoch_num = 20

    train_batch_size = batch_size
    val_batch_size = batch_size

    train_dataset = ImageFolder(data_root, mode='train')

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )
    for epoch in range(1, epoch_num + 1):
        st = time()
        dice_1 = 0.0
        dice_2 = 0.0
        dice_3 = 0.0
        dice_4 = 0.0
        d_len = 0
        for inputs, masks in tqdm(data_loader, ncols=50, total=len(data_loader)):
            x = inputs.cuda()
            y = masks.cuda()
            optimizer.zero_grad()
            output = transUnet(x)
            loss = criterion(output, y)
            # optimizer.step()
            output = torch.sigmoid(output)
            output[output > 0.5] = 1
            output[output < 0.5] = 0
            dice_Esophagus = diceCoeffv2(output[:, 0:1, :], y[:, 0:1, :], activation=None).cpu().item()
            dice_Heart = diceCoeffv2(output[:, 1:2, :], y[:, 1:2, :], activation=None).cpu().item()
            dice_Trachea = diceCoeffv2(output[:, 2:3, :], y[:, 2:3, :], activation=None).cpu().item()
            dice_Aorta = diceCoeffv2(output[:, 3:4, :], y[:, 3:4, :], activation=None).cpu().item()
            mean_dice = (dice_Esophagus + dice_Heart + dice_Trachea + dice_Aorta) / 4
            d_len += 1
            dice_1 += dice_Esophagus
            dice_2 += dice_Heart
            dice_3 += dice_Trachea
            dice_4 += dice_Aorta
            loss.backward()
            optimizer.step()
            iters += batch_size
            string_print = "Epoch = %d iters = %d Current_Loss = %.4f Mean Dice=%.4f Esophagus Dice=%.4f Heart " \
                           "Dice=%.4f Trachea Dice=%.4f Aorta Dice=%.4f Time = %.2f" \
                           % (epoch, iters, loss.item(), mean_dice,
                              dice_Esophagus, dice_Heart, dice_Trachea, dice_Aorta, time() - st)
            st = time()
            writer.add_scalar('train_loss', loss.item(), iters)
        dice_1 = dice_1 / d_len
        dice_2 = dice_2 / d_len
        dice_3 = dice_3 / d_len
        dice_4 = dice_4 / d_len
        m_dice = (dice_1 + dice_2 + dice_3 + dice_4) / 4
        print(
            'Epoch {}/{},Train Mean Dice {:.4}, Esophagus Dice {:.4}, Heart Dice {:.4}, Trachea Dice {:.4}, Aorta Dice {:.4}'.format(
                epoch, epoch_num, m_dice, dice_1, dice_2, dice_3, dice_4
            ))
        if epoch == epoch_num:
            torch.save(transUnet, 'D:/CS640-Project-TransUnet/TransUnet/checkpoint/halfdataset/{}.pth'.format(model_name + loss_name))
            writer.close()


def main():
    transUnet = get_transNet(n_classes=5).cuda()
    criterion = SoftDiceLossV2(activation='sigmoid').cuda()
    optimizer = optim.Adam(transUnet.parameters(), lr=1e-4)
    train(transUnet, criterion, optimizer, 0)


if __name__ == '__main__':
    main()
