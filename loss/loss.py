import torch.nn as nn
from torch.nn.modules.loss import _Loss
from TransUnet.loss.dice import *


class SoftDiceLossV2(_Loss):
    __name__ = 'dice_loss'

    def __init__(self, num_classes=5, activation='sigmoid', reduction='mean'):
        super(SoftDiceLossV2, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice