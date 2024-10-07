#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:07:32 2020

@author: cbrengman

从多个文件夹中读取图像文件，进行一些几何变换，然后保存到新的文件夹中
"""

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp, resize

fold = 'Transfer_Data/'
filel = ['val/data/', 'val/noise/', 'train/data/', 'train/noise/']  #原始图像文件夹

for folder in filel:
    fileloc = fold + folder
    filenames = [f for f in os.listdir(fileloc) if os.path.isfile(os.path.join(fileloc, f))]  #获取文件夹中的所有文件

    for image in filenames:
        img = io.imread(fileloc + image)
        # 如果图像不是 uint8 类型或范围不在 0-255，需要进行处理
        if img.dtype != np.uint8:
            if img.max() <= 1.0:  # 如果值在 0-1 之间，缩放到 0-255
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # 旋转角度设置
        rot_angles = [k for k in range(0, 360, 30)]

        # 翻转图像（左右、上下以及上下左右）
        flipped = [img, np.fliplr(img), np.flipud(img), np.flipud(np.fliplr(img))]

        # 将翻转后的图像进行旋转
        rotated = []
        for tmpimage in flipped:
            for angle in rot_angles:
                rotated.append(rotate(tmpimage, angle=angle, mode='constant', preserve_range=True))
        
        # 进行平移变换
        transform1 = AffineTransform(translation=(25, 25))
        transform2 = AffineTransform(translation=(-25, -25))
        shifted = []
        for tmpimage in rotated:
            shifted.append(warp(tmpimage, transform1, mode='constant', preserve_range=True))
            shifted.append(warp(tmpimage, transform2, mode='constant', preserve_range=True))
        
        # 将平移后的图像也添加到最终的旋转图像列表中
        for tmpimage in shifted:
            rotated.append(tmpimage)
        
        # 保存每张生成的图像
        for i, outimg in enumerate(rotated):
            # 确保每张图像在保存之前是 uint8 类型，并且值范围是 [0, 255]
            if outimg.max() <= 1.0:
                outimg = (outimg * 255).astype(np.uint8)
            else:
                outimg = outimg.astype(np.uint8)

            # 保存图像
            io.imsave('Transfer_Data/augmented/' + folder + image[:-4] + '_' + str(i) + '.png', outimg)