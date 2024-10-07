#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:57:42 2019

@author: cbrengman
"""

import time
import torch
from tqdm import tqdm  #进度条
from tkinter import Tk
from torch import nn,optim  #神经网络模块，优化器
from torchnet import meter  #评估模块  
from torchvision import datasets,transforms  #图像处理模块
from torch.utils.data import DataLoader  #数据加载模块 
from dataloaders.loaders import MyImageFolder  #自定义数据加载模块
from dataloaders.tvtsplits import Split90_10  #数据集划分模块
from tkinter.filedialog import askopenfilename
from models.CNN.DD.sarnet import sarnet1 as net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)      

def train_model(num_epochs=20,batch_size=20,load_checkpoint=False):
    """
    Initializes model. Loads data. Trains Model. 
    
    Args:
        num_epochs(int,optional): number of epochs to train for. default: 20
        load_checkpoint(bool,optional): whether or not to continue a previous training
    """
    
    
    #model = STsarnet_Classifier().to(device)
    model = net()
    model.to(device)
    
    
    loss_fn = nn.CrossEntropyLoss()    #交叉熵损失函数，常用于多分类问题
    optimizer = optim.SGD(model.parameters(),lr=0.01)  #随机梯度下降优化器，学习率为0.01
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)  #学习率调整器，每10个epoch学习率乘以0.1
    
    #Get dataloaders
    directory = "/home/zyh/SarNet/Synthetic_Data/data_smol/train"
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])  #图像处理，转换为灰度图像，转换为tensor，归一化
    
    #两种方法加载数据集
    #1：使用函数来划分训练集和验证集
    image_data=datasets.ImageFolder(directory,transform=transform)  #加载数据集
    train_dataloader, val_dataloader = Split90_10(image_data,batch_size)  #使用函数来划分训练集和验证集
    #2：直接读取文件夹内分割后的训练集和验证集
    # train_dataloader = DataLoader(MyImageFolder(directory,mode='train',transform=transform),batch_size=batch_size,shuffle=True)  #加载训练数据
    # val_dataloader = DataLoader(MyImageFolder(directory,mode='val',transform=transform),batch_size=batch_size*2,shuffle=True) #直接读取文件夹内分割后的验证集
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}  #数据加载器
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}  #数据集大小
    
    start = time.time()
    epoch_resume = 0
    best_acc = 0
    
    #构建混淆矩阵：是一种用于分类任务中评估模型性能的工具，通过矩阵形式展示模型的预测结果
    confusion_matrix = meter.ConfusionMeter(model.fc.out_features)
    
    if load_checkpoint:
        #Asks for filename and loads checkpoint model
        root = Tk()
        root.withdraw()
        file = askopenfilename()
        checkpoint = torch.load(file)
        print("Reloading from previously saved checkpoint")
        
        #Restores model state to model
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        #grabs the epoch to resume training
        epoch_resume = checkpoint["epoch"]
        best_acc = checkpoint["acc"]
        
    for epoch in tqdm(range(epoch_resume,num_epochs),unit="epochs",total=num_epochs,initial=epoch_resume):
        #Alternate between train and val phases
        for phase in ['train','val']:
            #Set loss and corrects for each epoch
            running_loss = 0.0
            running_corrects = 0
            
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
                
            for inputs,labels in dataloaders[phase]:
                #Move inputs to device
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                #keep grad to allow for backprop during training and disable during 
                #eval for faster evals
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = loss_fn(outputs,labels.long())
                    
                    #backprop during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.long())
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            confusion_matrix.add(outputs.data,labels.data)
            
            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")
            
        #Save the model if the test acc is greater than our current best
        if epoch_acc > best_acc:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': epoch_acc,
                    'opt_dict': optimizer.state_dict(),
                    }, "1ch_model_comb_e{}_Test.model".format(epoch+1))
            best_acc = epoch_acc
            
        time_elapsed = time.time() - start
        print(f"Training for epoch {epoch+1} completed in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")
    
        # Print the metrics
        print("Epoch %i, Train Accuracy: %.2f%% , TrainLoss: %.2f%%" % (epoch, epoch_acc, epoch_loss))
        if model.fc.out_features > 1:
            print("Confusion Matrix: ")
            print("[[TN,FP]" + '\n' + "[FN,TP]]")
            print(confusion_matrix.conf)
    
    time_elapsed = time.time() - start
    print(f"Training completed in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")
    print(f"Best model accuracy: {best_acc}")
        
if __name__ == "__main__":
    train_model(num_epochs=15,load_checkpoint=False)