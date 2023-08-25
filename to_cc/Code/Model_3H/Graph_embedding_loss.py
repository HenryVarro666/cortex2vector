import torch
import torch.nn as nn


def mse(target, predict, hot_num):
    loss = 0
    for hot in range(hot_num):
        loss += torch.nn.functional.mse_loss(predict[:, hot, :], target[:, hot, :])
    return loss
