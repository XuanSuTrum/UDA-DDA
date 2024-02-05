# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:23:16 2023
@author: Administrator
"""
#在models_mmd_cmmd中加入softmax
import argparse
import os
import SDA_DDA
import utils
import numpy as np
import torch
from get_dataset import get_dataset
from load_data2 import load_data
import math
import torch.nn as nn
import random

import torch.nn.functional as F
from torch.nn import init
import pandas as pd
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_rows', None)#显示全部行
pd.set_option('display.max_columns', None)#显示全部列
np.set_printoptions(threshold=np.inf)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log = []

# Command setting
parser = argparse.ArgumentParser(description='DDC_DCORAL')
parser.add_argument('--model', type=str, default='simple_net')
parser.add_argument('--batchsize', type=int, default=256)
# parser.add_argument('--src', type=str, default='amazon')
# parser.add_argument('--tar', type=str, default='webcam')
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--lamb', type=float, default=0.5)
parser.add_argument('--trans_loss', type=str, default='mmd')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')
args = parser.parse_args(args=[])

def setup_seed(seed): ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def weigth_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
#        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()

def segmented_function(epoch):
    if epoch <= 10:
        value = 1
    elif 10 < epoch <=40:
        # 在10-30之间逐渐减小值，你可以根据需要调整
        value = 2/ (1 + math.exp(-10 * (args.n_epoch) / args.n_epoch)) - 1
    elif 40 < epoch <= 85:
        # 在10-30之间逐渐减小值，你可以根据需要调整
        value = 1 * np.exp(-0.6 * epoch)
    else:
        value = 0

    return value

def segmented_function_1(epoch):
    if epoch <= 40:
        value = 0.65
    else:
        value = 1

    return value

def tt(model, target_test_loader):
    model.eval()
    correct = 0

    len_target_dataset = len(target_test_loader.dataset)

    num_classes = 3
    conf_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for data, target in target_test_loader:


            data, target = data.to(DEVICE), target.to(DEVICE)

            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]

            target = torch.argmax(target, dim=1)
            correct += torch.sum(pred == target)
            conf_matrix += confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=np.arange(num_classes))

    acc = 100. * correct / len_target_dataset
    return acc, pred,conf_matrix

def train(source_loader, target_train_loader, target_test_loader, model, optimizer):
    len_source_loader = len(source_loader)  # 1106
    len_target_loader = len(target_train_loader)  # 53
    best_acc = 0
    stop = 0
    best_confusion_matrix = None
    for e in range(args.n_epoch):
        # print('e',e)
        data_target_ = []
        data_source_ = []
        t_label_s = []
        s_label_s = []
        all_pseudo_labels = []
        stop += 1
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_cmmd_loss = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)

        criterion = torch.nn.CrossEntropyLoss()
        for mlen in range(n_batch):

            # print('mlen',mlen)
            data_source, label_source = iter_source.next()
            data_target, label_target = iter_target.next()
            if mlen % len(target_train_loader) == 0:
                iter_target = iter(target_train_loader)
            if cuda:
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target, label_target = data_target.cuda(),label_target.cuda()

            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target, label_target = data_target.to(DEVICE), label_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss, cmmd_loss = model(e,data_source, data_target,label_source)

            clf_loss = criterion(label_source_pred, label_source)


            if args.gamma == 1:
                gamma = 2 / (1 + math.exp(-10 * (args.n_epoch) / args.n_epoch)) - 1
            if args.gamma == 2:
                gamma = args.n_epoch / args.n_epoch


            beta = segmented_function(e)
            beta_1 = segmented_function_1(e)

            # 添加条件判断，如果clf_loss低于0.4，cmmd_loss前的权重为1，否则为0.1
            if clf_loss <= 0.1:
                cmmd_weight = 1.0
            elif 0.1 < clf_loss < 0.15:
                cmmd_weight = 0.5
            else:
                cmmd_weight = 0

            loss = clf_loss + beta*transfer_loss+ cmmd_weight*cmmd_loss

            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_cmmd_loss.update(cmmd_loss.item())
            train_loss_total.update(loss.item())

            data_source_.append(data_source)
            data_target_.append(data_target)

            s_label_s.append(label_source)
            t_label_s.append(label_target)

        # Test
        acc,pred,conf_matrix = tt(model, target_test_loader)
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_cmmd_loss.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt('F:\\Emotion_datasets\\SEED\\train_log.csv', np_log, delimiter=',', fmt='%.6f')
        print('Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, train_loss_cmmd_loss: {:.4f},total_Loss: {:.4f}, acc: {:.4f}'.format(
                 e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_cmmd_loss.avg,
                 train_loss_total.avg, acc))
        if best_acc < acc:
            best_acc = acc
            best_confusion_matrix = conf_matrix

    print('Transfer result: {:.4f}'.format(best_acc))
    print(best_confusion_matrix)
    # 在每个epoch结束时，将伪标签追加到 all_epochs_pseudo_labels 中

    # np.save('all_epochs_pseudo_labels.npy', np.array(all_epochs_pseudo_labels))
    return best_acc

if __name__ == '__main__':
    #setup_seed(20)
    all_test_results = []
    all_epochs_pseudo_labels = []
    for i in range(15):
        torch.manual_seed(0)
        #setup_seed(20)
        SESSION = 1
        batch_size = 32
        print('test_id = ', i + 1)
        #source_loader, target_train_loader, target_test_loader = load_data(test_id=i, session=SESSION,BATCH_SIZE=batch_size)
        # source_loader, target_train_loader, target_test_loader = load_data(test_id=i, BATCH_SIZE=batch_size,session=SESSION)
        source_loader, target_train_loader, target_test_loader = load_data(test_id=i, BATCH_SIZE=batch_size,session=SESSION)
        target_set, source_set = get_dataset(test_id=i, session=SESSION)
        source_features, source_labels = torch.from_numpy(source_set['feature']), torch.from_numpy(source_set['label'])
        test_features, test_labels = torch.from_numpy(target_set['feature']), torch.from_numpy(target_set['label'])
        #setup_seed(20)
        model = SDA_DDA.Transfer_Net(
            args.n_class, transfer_loss=args.trans_loss, base_net=args.model).to(DEVICE)

        optimizer = torch.optim.Adam([
            {'params': model.base_network.parameters()}], lr=args.lr)

        model.apply(weigth_init)
        transfer_results = train(source_loader, target_train_loader, target_test_loader, model, optimizer)
        all_test_results.append(transfer_results)

        model_filename = f'E:\\CODE\\transfer_learning\\DDC_EEGNet\\model_parameters\\main_mmd_cmmd_1\\model_parametersmodel_subject_{i}.pth'
        torch.save(model.state_dict(), model_filename)
    # 将所有测试结果堆叠成一个NumPy数组
    stacked_results = torch.stack(all_test_results)

    # 计算所有测试结果的平均值
    average_result = torch.mean(stacked_results)

    # 打印平均结果
    print("Average result:", average_result)