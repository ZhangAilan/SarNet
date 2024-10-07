# -*- coding: utf-8 -*-
'''
ailanzhang1109@gmail.com
2024-9-21

提取卷积层的滤波器并显示
'''

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import os
import sys
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # SarNet/Net 目录
sys.path.append(project_root)
from models.CNN.DD.sarnet import sarnet1 as net

def show_images(images, rows=1):
    """Display a list of images in a single figure with matplotlib."""
    n_images = len(images)
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(rows, int(np.ceil(n_images / float(rows))), n + 1)
        a.set_xticks([])
        a.set_yticks([])
        plt.imshow(image, cmap='jet')  # 使用 'jet' colormap 显示滤波器
    fig.set_size_inches(figsize)
    plt.tight_layout(pad=1.0)
    plt.show()

def main():
    # 设置模型路径、卷积层索引和行数
    model_path = '/home/zyh/SarNet/Net/models_out/SarNet.model'  # 替换为你的模型路径
    layer_index = 0  # 替换为要提取的卷积层索引
    rows = 4  # 显示的行数

    # 加载模型
    model = net()  # 创建模型
    checkpoint = torch.load(model_path)  # 加载检查点
    model.load_state_dict(checkpoint['state_dict'])  # 提取模型的状态字典
    model.eval()  # 设置模型为评估模式

    # 提取模型中的卷积层
    conv_layers = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append(m)

    # 提取卷积核的滤波器（选择第 layer_index 个卷积层）
    filters = conv_layers[layer_index].weight.data.cpu().numpy()  # 提取卷积层的权重，并转换为 numpy 数组

    # 提取卷积核的第一个通道（通常是灰度图）
    img2 = []
    for filt in filters:
        img2.append(filt[0])  # 提取每个卷积核的第一个通道

    # 定义图像的显示尺寸
    global figsize
    figsize = [13.25, 7.5]

    # 显示提取出的滤波器图像
    show_images(img2, rows=rows)  # 将图像按行显示，调整行数使得图像合理分布

if __name__ == "__main__":
    main()