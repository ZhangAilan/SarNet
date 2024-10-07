# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:44:22 2019

@author: cbrengman

加载图像并对其进行预测，判断图像中的内容是否为“”数据“”或“”噪声“”
"""

# Import needed packages
import torch
from torchvision.transforms import transforms  #torchvision.transforms模块提供了一般的图像转换操作类
from torch.autograd import Variable #torch.autograd.Variable类对Tensor对象进行封装，主要用于神经网络的反向传播
import os
import sys
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # SarNet/Net 目录
sys.path.append(project_root)

from models.CNN.DD.sarnet import sarnet1 as net #导入模型
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image #PIL是Python Imaging Library，是Python平台事实上的图像处理标准库
from misc.slice_join_image import slice_image, join_image


checkpoint = torch.load("/home/zyh/SarNet/Net/models_out/SarNet.model")  #加载模型
print(checkpoint.keys())
model = net(pretrained=True)
model.load_state_dict(checkpoint['state_dict'])  #加载模型参数
model.eval()

def predict_image(img):
    print("Prediction in progress")
    
    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    # Preprocess the image
    image_tensor = transformation(img).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    # Turn the input into a Variable
    image = Variable(image_tensor)

    # Predict the class of the image
    output = model(image)

    index = output.data.numpy().argmax()

    return index

def predict_images(img):
    print("Prediction in progress")
    
    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    
    index = []
    # Preprocess the image
    for im in img:
        image_tensor = transformation(im.image).float().unsqueeze_(0)

        # Turn the input into a Variable
        image = Variable(image_tensor)

        # Predict the class of the image
        output = model(image)

        index.append(output.data.numpy().argmax())
        
        #fig,ax = plt.subplots()
        #ax.imshow(np.asarray(im.image),cmap='gray')
        #if output.data.numpy().argmax()==0:
        #    ax.set_title("Class = Noise")
        #else:
        #    ax.set_title("Class = Data")

    return index

def load_image(filename,size=(224,224),op='downsample'):
    """
    Load an Image and
    Split an image into N smaller images of size (tuple)
    or downsample to size
    
    Args:
        filename (str): Filename of the image to split/downsample
        size (tuple): the size of the smaller images
        
    Kwargs:
        op (str): operation (downsample or split)
        
    returns:
        tuple of :class:`tile` instances
    """
    
    if op == "downsample":
        img = Image.open(filename).convert('L')
        img = img.resize(size, Image.Resampling.LANCZOS)
    elif op == "slice":
        img = slice_image(filename,size=size)
    else:
        raise Exception("Invalid option '{}'. Valid options are 'downsample' or 'slice'.".format(op))
    
    return img
    

if __name__ == "__main__":
    filename='/home/zyh/SarNet/Net/Transfer_Data/train/noise/8.png'
    size = (224,224)

    # load image
    img = load_image(filename,size=size,op="downsample")
    # img = load_image(filename,size=size,op="slice")
    
    #check if one or many images
    if hasattr(img,'__len__'):
        index = predict_images(img)
        for im,val in zip(img,index):
            im.score = val
        image,score = join_image(img)
        fig,ax = plt.subplots()
        ax.imshow(image,cmap='gray')
        sc = ax.imshow(score,alpha = 0.25,cmap='jet_r')
        ax.set_xticks(np.arange(0,image.size[0]+1,size[0]))
        ax.set_yticks(np.arange(0,image.size[1]+1,size[1]))
        ax.grid(color='k',linestyle='-',linewidth=2)
        cbar = fig.colorbar(sc,ticks=[0,255])
        cbar.ax.set_yticklabels(['Data','Noise'])
        plt.show()
    else:
        index = predict_image(img)
        fig,ax = plt.subplots()
        ax.imshow(np.asarray(img),cmap='gray')
        if index==0:
            ax.set_title("Class = Noise (0)")
        else:
            ax.set_title("Class = Data (1)")
        plt.show()