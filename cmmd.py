# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:14:18 2023

@author: Administrator
"""

import torch
import numpy as np
from torch.autograd import Variable

min_var_est = 1e-8

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)
def cmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    s_label = s_label.cpu()

    # s_label = torch.argmax(s_label).item()
    # s_label = [s_label.index(1) for one_hot in s_label]
    s_label = s_label.view(-1, 1)
    s_label = s_label.to(torch.int64)
    s_label = torch.zeros(s_label.shape[0], 3).scatter_(1, s_label.data, 1)
    s_label = Variable(s_label).cuda()

    t_label = t_label.cpu()
    t_label = t_label.view(-1, 1)
    t_label = torch.zeros(t_label.shape[0], 3).scatter_(1, t_label.data, 1)
    t_label = Variable(t_label).cuda()

    batch_size_s = int(s_label.size()[0])
    batch_size_t = int(t_label.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    # 提取相似性矩阵
    XX = kernels[:batch_size_s, :batch_size_s]

    # 检查目标域是否为空
    if batch_size_t == 0:
        # 如果目标域为空，直接计算源域的损失
        loss = torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX)
    else:
        # 如果目标域不为空，按照之前的流程进行计算
        YY = kernels[batch_size_s:, batch_size_s:]
        XY = kernels[:batch_size_s, batch_size_s:]
        YX = kernels[batch_size_s:, :batch_size_s]

        loss_XX = torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX)
        loss_YY = torch.mean(torch.mm(t_label, torch.transpose(t_label, 0, 1)) * YY)
        loss_XY = torch.mean(torch.mm(s_label, torch.transpose(t_label, 0, 1)) * XY)
        loss_YX = torch.mean(torch.mm(t_label, torch.transpose(s_label, 0, 1)) * YX)
        
        loss += loss_XX + loss_YY - loss_XY - loss_YX

    loss /= 3

    return loss


# # 随机生成源域数据和目标域数据
# source_data = torch.randn(10, 50)  # 假设源域数据维度为 (100, 50)
# target_data = torch.randn(10, 50)   # 假设目标域数据维度为 (50, 50)
#
# # 随机生成源域真实标签和目标域伪标签
# source_labels = torch.randint(0, 2, (10,))  # 二分类任务
# target_pseudo_labels = torch.randint(0, 2, (10,))  # 随机生成目标域伪标签
#
# # 随机生成置信度，这里假设置信度在 [0, 1] 之间
# confidence_threshold = torch.tensor([0.99])
#
# # 置信度判断，选择高于置信度阈值的目标域数据
# confidence_mask = torch.rand(target_data.size(0)) > confidence_threshold
# confident_target_data = target_data[confidence_mask]
# confident_target_pseudo_labels = target_pseudo_labels[confidence_mask]
# source, target, s_label, t_label = source_data, confident_target_data, source_labels, confident_target_pseudo_labels
# kernel_mul=2.0
# kernel_num=5
# fix_sigma=None
#
# loss = cmmd(source_data, confident_target_data, source_labels, confident_target_pseudo_labels)
#
# print("CMMD Loss:", loss.item())
