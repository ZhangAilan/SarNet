# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:03:21 2019

@author: cbrengman

加载图像，生成类激活图
"""

import cv2
import torch 
from torch import optim
import numpy as np
from PIL import Image
from tkinter import Tk
import tifffile as tiff
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from models.CNN.DD.sarnet import sarnet1
from tkinter.filedialog import askopenfilename
from mpl_toolkits.axes_grid1 import make_axes_locatable
from misc.slice_join_image import slice_image, join_image_heat


model = sarnet1()
optimizer = optim.SGD(model.parameters(),lr=0.01)
#Asks for filename and loads checkpoint model
root = Tk()
root.withdraw()
file = askopenfilename()
checkpoint = torch.load(file)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['opt_dict'])
finalconv_name = 'layer4'
net = model
net.eval()

    
def gen_heat(image):

    # hook the feature extractor，在前向传播时捕获特征层提取层的输出（即中间层的特征图）
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    #获取网络模型的最后一层卷积层   
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
     
    # get the softmax weight：接近输出层的权重
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 224x224
        size_upsample = (224, 224)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        output_cam_og = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample,interpolation=cv2.INTER_CUBIC))
            output_cam_og.append(cam_img)
        return output_cam,output_cam_og
    
    preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    
    img_tensor = preprocess(image)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if img_variable.shape[1] > 2:
        img_variable = img_variable.transpose(1,2)
    logit = net(img_variable)
    
    classes = {0: 'data',1: 'noise'}
    
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    
    # output the prediction
    for i in range(2):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    out = []
    out.append(classes[idx[0]])
    out.append(classes[idx[1]])
    
    # generate class activation mapping for the top1 prediction
    CAMs,OGCAMs = returnCAM(features_blobs[0], weight_softmax, idx)
    
    height,width = image.size
    #CAMs[0] = cv2.resize(CAMs[0],(width,height))
    #CAMs[1] = cv2.resize(CAMs[1],(width,height))
    return CAMs,OGCAMs,out
    




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
    
    try:
        img = Image.open(filename).convert('L')
    except:
        print("PIL can't open image. Trying tiff.imread")
        try:
            img = tiff.imread(filename)
            if len(img) >= 1:
                img[0] = cv2.normalize(img[0],None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                img[1] = cv2.normalize(img[1],None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                img1 = Image.fromarray(img[0])
                img2 = Image.fromarray(img[1])
                img1 = img1.convert('L')
                img2 = img2.convert('L')
                img3 = Image.new('L',img1.size)
                img = Image.merge('RGB',[img1,img2,img3])
            else:
                img = cv2.normalize(img,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                img = Image.fromarray(img,'L')
            print('Image file loaded correctly')
        except:
            print('Cannot Load Image File')   
    if op == "downsample":
        img = img.resize(size, Image.Resampling.LANCZOS)
    elif op == "slice":
        img = slice_image(img,size=size)
    else:
        raise Exception("Invalid option '{}'. Valid options are 'downsample' or 'slice'.".format(op))
    
    return img


if __name__ == "__main__":
    mode = 'downsample'  #downsample or slice
    # mode = 'slice'
    # filename = '/home/zyh/SarNet/Net/Transfer_Data/val/data/19.png'
    # filename = '/home/zyh/SarNet/Real_Data/20170404-20180716.png'
    # filename = '/home/zyh/SarNet/Real_Data/sys_20170416-20180821.png'
    filename = '/home/zyh/SarNet/Real_Data/59ad8c335c597032726ce5ead7f0a648226953ba.jpg'
    size = (224,224)
    img = load_image(filename,size,mode)
    
    #check if one or many images
    if hasattr(img,'__len__'):
        CAMs = []
        sco = []
        for i,im in enumerate(img):
            CAM,_,sc = gen_heat(im.image)
            CAMs.append(CAM)
            if sc == 'data':
                sco.append(1)
            elif sc =='noise':
                sco.append(0)
        for im,cam,sc in zip(img,CAMs,sco):
            im.cam = cam
            im.score = sc
        image,score,heat = join_image_heat(img)   #此处仍有错误，无法将cam属性传入
        
        fig,ax = plt.subplots()
        ax.imshow(image,cmap='gray')
        sc = ax.imshow(score,alpha = 0.25,cmap='jet')
        ax.set_xticks(np.arange(0,image.size[0]+1,size[0]))
        ax.set_yticks(np.arange(0,image.size[1]+1,size[1]))
        ax.grid(color='k',linestyle='-',linewidth=2)
        cbar = fig.colorbar(sc,ticks=[0,255])
        cbar.ax.set_yticklabels(['Noise','Data'])
        
        fig1,ax1 = plt.subplots()
        ax1.imshow(image,cmap='gray')
        sc = ax1.imshow(heat,alpha = 0.25,cmap='jet')
        ax1.set_xticks(np.arange(0,image.size[0]+1,size[0]))
        ax1.set_yticks(np.arange(0,image.size[1]+1,size[1]))
        ax1.grid(color='k',linestyle='-',linewidth=2)
        cbar = fig1.colorbar(sc,ticks=[0,255])
        cbar.ax.set_yticklabels(['Low Activation','High Activation'])
        plt.show()
    else:
        if len(img.getbands()) > 1:
            img1 = np.array((np.array(img.getchannel(0)),np.array(img.getchannel(1))))
            img = img1
        CAMs,OGCAMs,classification = gen_heat(img)
        #plot
        fig,ax = plt.subplots(1,3)
        im = ax[0].imshow(img,cmap='gray')
        ax[0].title.set_text('Input Image')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im,cax=cax)
        im1 = ax[1].imshow(CAMs[1],cmap='jet')
        ax[1].title.set_text('Class Activation Map: ' + classification[1])
        ax[1].set_yticklabels('')
        divider = make_axes_locatable(ax[1])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1,cax=cax1)
        im2 = ax[2].imshow(img,cmap='gray')
        im2 = ax[2].imshow(CAMs[1],alpha=0.4,cmap='jet')
        ax[2].title.set_text('Overlay')
        ax[2].set_yticklabels('')
        divider = make_axes_locatable(ax[2])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2,cax=cax2)
        im2.set_clim(0,255)
        #fig.suptitle("Image '" + fname + "' Heatmap\nClassification: " + classification,y=0.8)
        plt.show()