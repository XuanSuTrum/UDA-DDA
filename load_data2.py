# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:53:27 2023

@author: Administrator
"""
import numpy as np
import torch.utils.data as Data
import torch
#import ctypes
#ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import random
import matplotlib.pyplot as plt
from torch.optim import Adam,SGD,RMSprop
from typing import Optional
import scipy.io as scio
from torch.optim.optimizer import Optimizer
from sklearn import preprocessing
import torch.nn.functional as F
def load_data(test_id,session,BATCH_SIZE):
    
    path = 'F:\\Emotion_datasets\\SEED\\feature_for_net_session'+str(session)+'_LDS_de'
    os.chdir(path)
#    path='F:\\zhourushuang\\transfer_learning\\feature_for_net_session1_LDS_de'
    #path = 'F:\Emotion_datasets\SEED\feature_for_net_session1_LDS_de'
    feature_list=[]
    label_list=[]
    ## our label:0 negative, label:1 :neural,label:2:positive, seed original label: -1,0,1, our label= seed label+1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain,info)
        if session==1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0,0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0,0]
        elif session==2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0,0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0,0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0,0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0,0]
        #feature = feature.reshape(842,-1)
        feature_list.append(min_max_scaler.fit_transform(feature).astype('float32')) # Variable 'feature' is a [3394, 310] DE feature matrix from SEED dataset.
        one_hot_label_mat=np.zeros((len(label),3)) # Variable 'one_hot_label_mat' is a [3394, 3] ground-truth matrix from SEED dataset.
        for i in range(len(label)):
            if label[i]==-1: # '0' refers to '-1 (negative emotion)' in SEED 
                one_hot_label=[1,0,0]
                one_hot_label=np.hstack(one_hot_label).reshape(1,3)
                one_hot_label_mat[i,:]=one_hot_label
            if label[i]==0: # '1' refers to '0 (neutral emotion)' in SEED 
                one_hot_label=[0,1,0]
                one_hot_label=np.hstack(one_hot_label).reshape(1,3)
                one_hot_label_mat[i,:]=one_hot_label
            if label[i]==1: # '2' refers to '1 (positive emotion)' in SEED 
                one_hot_label=[0,0,1]
                one_hot_label=np.hstack(one_hot_label).reshape(1,3)
                one_hot_label_mat[i,:]=one_hot_label
        label_list.append(one_hot_label_mat.astype('float32'))
    target_feature,target_label=feature_list[test_id],label_list[test_id]
    #target_feature = target_feature.reshape(842,1,62,800)
    del feature_list[test_id]
    del label_list[test_id]
    source_data,source_label=np.vstack(feature_list),np.vstack(label_list)
    #source_feature = source_feature.reshape(source_feature.shape[0],1,62,800)
    print('target_feaure.shape',target_feature.shape)
    target_train_data,target_train_label = target_feature[:,:],target_label[:,:]
    target_test_data,target_test_label = target_feature[:,:],target_label[:,:]

    torch_dataset_source = Data.TensorDataset(torch.from_numpy(source_data),torch.from_numpy(source_label))
    torch_dataset_target_train = Data.TensorDataset(torch.from_numpy(target_train_data),torch.from_numpy(target_train_label))
    torch_dataset_target_test = Data.TensorDataset(torch.from_numpy(target_test_data),torch.from_numpy(target_test_label))

    source_loader = Data.DataLoader(
             dataset=torch_dataset_source,
             batch_size=BATCH_SIZE,
             shuffle=True,
             num_workers=0,
             drop_last=True
             )
    target_train_loader = Data.DataLoader(
             dataset=torch_dataset_target_train,
             batch_size=BATCH_SIZE,
             shuffle=True,
             num_workers=0,
             drop_last=True
             )   
    target_test_loader = Data.DataLoader(
             dataset=torch_dataset_target_test,
             batch_size=target_test_data.shape[0],
             shuffle=True,
             num_workers=0,
             # drop_last=True
             )
    return source_loader, target_train_loader, target_test_loader