# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:38:22 2023

@author: Administrator
"""
import numpy as np
import torch
# import ctypes
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import random
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD, RMSprop
from typing import Optional
import scipy.io as scio
from torch.optim.optimizer import Optimizer
from sklearn import preprocessing
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):  ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(test_id,
                session):  ## dataloading function, you should modify this function according to your environment setting.
    # path='F:\\zhourushuang\\transfer_learning\\feature_for_net_session'+str(session)+'_LDS_de'
    # path = 'F:\Emotion_datasets\preprocess_data\800ms\\session'+str(session)
    path = 'F:\\Emotion_datasets\\SEED\\feature_for_net_session' + str(session) + '_LDS_de'
    os.chdir(path)
    #    path='F:\\zhourushuang\\transfer_learning\\feature_for_net_session1_LDS_de'
    # path = 'F:\Emotion_datasets\SEED\feature_for_net_session1_LDS_de'
    feature_list = []
    label_list = []
    ## our label:0 negative, label:1 :neural,label:2:positive, seed original label: -1,0,1, our label= seed label+1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)
        if session == 1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
        elif session == 2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0, 0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0, 0]
        # feature = feature.reshape(842,-1)
        feature_list.append(min_max_scaler.fit_transform(feature).astype(
            'float32'))  # Variable 'feature' is a [3394, 310] DE feature matrix from SEED dataset.
        one_hot_label_mat = np.zeros(
            (len(label), 3))  # Variable 'one_hot_label_mat' is a [3394, 3] ground-truth matrix from SEED dataset.
        for i in range(len(label)):
            if label[i] == -1:  # '0' refers to '-1 (negative emotion)' in SEED
                one_hot_label = [1, 0, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 0:  # '1' refers to '0 (neutral emotion)' in SEED
                one_hot_label = [0, 1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 1:  # '2' refers to '1 (positive emotion)' in SEED
                one_hot_label = [0, 0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
        label_list.append(one_hot_label_mat.astype('float32'))
    target_feature, target_label = feature_list[test_id], label_list[test_id]
    # target_feature = target_feature.reshape(842,1,62,800)
    del feature_list[test_id]
    del label_list[test_id]
    source_feature, source_label = np.vstack(feature_list), np.vstack(label_list)
    # source_feature = source_feature.reshape(source_feature.shape[0],1,62,800)

    target_set = {'feature': target_feature, 'label': target_label}
    source_set = {'feature': source_feature, 'label': source_label}
    return target_set, source_set


class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 62),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(62, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((400), classes_num)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1), x  # return x for visualization


def train_and_test_GAN(test_id, session, threshold_update=True):  ## pipeline for PR-PL model
    setup_seed(20)
    BATCH_SIZE = 96
    ## dataloader(test_id: test subject in the LOOCV process, session:1,2,3 for different sessions in the SEED dataset)
    target_set, source_set = get_dataset(test_id, session)
    torch_dataset_train = Data.TensorDataset(torch.from_numpy(source_set['feature']),
                                             torch.from_numpy(source_set['label']))
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature']),
                                            torch.from_numpy(target_set['label']))

    test_features, test_labels = torch.from_numpy(target_set['feature']), torch.from_numpy(target_set['label'])
    source_features, source_labels = torch.from_numpy(source_set['feature']), torch.from_numpy(source_set['label'])

    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=test_features.shape[0],
        shuffle=True,
        num_workers=0
    )
    setup_seed(20)

    model = EEGNet(classes_num=3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    T = source_features.shape[0] // BATCH_SIZE
    model.train()
    epoch_losses = []
    epoch_accuracies = []

    for i in range(T):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for step, (b_x, b_y) in enumerate(loader_train):
            b_y = b_y.float()

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output, _ = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #     total_loss += loss.item()

        #     b_y = torch.argmax(b_y, dim=1)

        #     _, predicted = output.max(1)
        #     total += b_y.size(0)
        #     correct += predicted.eq(b_y).sum().item()

        # # 计算并存储每个epoch的平均损失和准确率
        # epoch_loss = total_loss / (step + 1)
        # epoch_accuracy = 100 * correct / total
        # epoch_losses.append(epoch_loss)
        # epoch_accuracies.append(epoch_accuracy)

        # print(f"Epoch [{i + 1}/{T}]: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")

    model.eval()
    for step, (test_x, test_y) in enumerate(loader_test):
        test_x = test_x.to(device)
        test_y = test_y.to(device)

        test_x = test_x.to(torch.float32)
        test_y = test_y.float()

        output_pre, feature_test = model(test_x)
        loss_pre = loss_func(output_pre, test_y)
        _, pred_y = torch.max(output_pre.data, 1)

        test_y = torch.argmax(test_y, dim=1)
        correct = (pred_y == test_y).sum()
        total = test_y.size(0)
        # print(feature_test)
        print('Epoch: ', step, '| test loss: %.4f' % loss.data.cpu().numpy(),
              '| test accuracy: %.2f' % (float(correct) / total))


if __name__ == '__main__':
    for i in range(15):
        print('test_id = ', i + 1)
        train_and_test_GAN(test_id=i, session=1, threshold_update=True)