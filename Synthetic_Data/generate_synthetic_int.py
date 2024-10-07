#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:35:07 2019

@author: Glarus
"""

from genrand_synth_displacements_smol import get_fault_parameters
from generate_synthetic_noise import gen_noise_corr_dem
import numpy as np
import cv2
from PIL import Image
import multiprocessing as mp
import matplotlib.pyplot as plt
import os

def get_data_noise():
    data = get_fault_parameters()  #返回形变
    n1,n2,n3 = gen_noise_corr_dem()  #返回三个噪声干扰的dem
    return data,n1,n2,n3

def wrap_images(img):
    img_wrapped = (((img - img.min()) * 4 * np.pi / 0.0555) % (2 * np.pi)) / 2 / np.pi
    return img_wrapped

def comb_data_noise(data,n1,n2,n3):
    wdata = wrap_images(data)
    wn1   = wrap_images(n1)
    wn2   = wrap_images(n2)
    wn3   = wrap_images(n3)
    
    wndata = (wdata+wn1)-wn2
    wndata = wndata - wndata.mean()
    wndata = wndata / max(abs(wndata.min()),abs(wndata.max()))
    
    ndata = (data+n1)-n2
    noise = n3
    nwdata = wrap_images(ndata) #(wdata+wn1)-wn2
    wnoise = wn3
    return ndata,noise,nwdata,wnoise

def make_greyscale(img):
    #将输入图像转换为灰度图像，并将像素值归一化到0-255
    img = cv2.normalize(img,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
    img = Image.fromarray(img,'L')
    return img
    
def save_images(x,loc,ndatag,noiseg,ndata_wrapg,noise_wrapg,datag,data_wrapg):
    # 定义需要保存的文件夹路径
    data_path = f"data_smol/{loc}/data/"
    noise_path = f"data_smol/{loc}/noise/"
    
    # 确保所有路径存在，如果不存在则创建
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(noise_path, exist_ok=True)

    ndata_wrapg.save("data_smol/" + loc + "/data/data_wrap_image" + str(x) + ".tif")
    Image.Image.close(ndata_wrapg)
    noise_wrapg.save("data_smol/" + loc + "/noise/noise_wrap_image" + str(x) + ".tif")
    Image.Image.close(noise_wrapg)

    
def main_loop(x,loc='train'):
    data,n1,n2,n3 = get_data_noise()
    ndata,noise,ndata_wrap,noise_wrap = comb_data_noise(data,n1,n2,n3)
    ndatag = make_greyscale(ndata)
    noiseg = make_greyscale(noise)
    datag = make_greyscale(data)
    # ndatag.save('noise_data.tif') #形变+噪声 
    # noiseg.save('noise.tif') #噪声 
    # datag.save('data.tif') #形变
    
    data_wrapg = make_greyscale(wrap_images(data))
    ndata_wrapg = make_greyscale(ndata_wrap)
    noise_wrapg = make_greyscale(noise_wrap)
    save_images(x,loc,ndatag,noiseg,ndata_wrapg,noise_wrapg,datag,data_wrapg)
    

if __name__ == "__main__":
    with mp.Pool(mp.cpu_count()) as pool:
        for proc in range(100000):
            pool.starmap(main_loop,[(proc,'train')])
    pool.close()