# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:22:18 2019

@author: cbrengman
"""

import torch.nn as nn  #构建神经网络，卷积层，全连接层等
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding
    创建3x3卷积层
    in_planes:输入通道数，out_planes:输出通道数，stride:步长
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1   #特征图的扩展因子，这里为1,表示没有特征图通道数的扩展

    def __init__(self, inplanes, planes, stride=1, downsample=None):  
        '''
        inplanes:输入通道数
        planes:输出通道数
        stride:步长
        downsample:下采样
        '''
        super(BasicBlock, self).__init__()  #调用父类的构造函数
        self.conv1 = conv3x3(inplanes, planes, stride)  #创建第一个3x3卷积层
        self.bn1 = nn.BatchNorm2d(planes)  #对卷积层的输出进行归一化处理
        self.relu = nn.ReLU(inplace=True)  #激活函数
        self.conv2 = conv3x3(planes, planes)  #创建第二个3x3卷积层
        self.bn2 = nn.BatchNorm2d(planes)  #对卷积层的输出进行归一化处理
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  #前向传播
        '''
        定义数据流动和计算过程的函数，通过接受输入数据并经过各个网络层的计算最终生成输出
        每个神经网络都要定义一个forward函数来描述输入数据如何经过层与层之间的计算过程
        '''
        residual = x  #保存输入x作为残差

        out = self.conv1(x)  #第一个卷积层输入x
        out = self.bn1(out)    #归一化
        out = self.relu(out)   #激活函数

        out = self.conv2(out)  #第二个卷积层
        out = self.bn2(out)  #归一化

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual   #残差连接
        out = self.relu(out)

        return out
    
class SarNet1ch(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64   #定义输入通道数为64
        super(SarNet1ch, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  #定义第一个卷积层，通道数为1（单通道图像），输出通道数为64，卷积核大小为7x7，步长为2，填充为3
        self.bn1 = nn.BatchNorm2d(64)  #对卷积层的输出进行归一化处理
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #最大池化方法，选择池化窗口中的最大值
        self.layer1 = self._make_layer(block, 64, layers[0]) #构建第一个残差块
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  #使用自适应平均池化方法，将任意大小的输入映射到固定大小的输出，输出大小为1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  #全连接层，输入通道数为512*block.expansion，输出通道数为num_classes，将特征图映射到类别空间

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  #判断m是否为卷积层
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  #计算卷积核的参数个数，总参数个数=卷积核大小*输出通道数
                #He/Kaiming初始化，适用于ReLU激活函数，能够在深度网络有效保持激活值的分布，助于防止梯度消失或梯度爆炸
                m.weight.data.normal_(0, math.sqrt(2. / n))  #使用正态分布初始化卷积层的权重，均值为0，标准差为math.sqrt(2. / n)
            elif isinstance(m, nn.BatchNorm2d):  #判断m是否为归一化层
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        创建多个残差块，在神经网络中堆叠多个残差块，构建深度网络
        block:残差块
        planes:输出通道数
        blocks:残差块的数量
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class SarNet2ch(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(SarNet2ch, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SarNet3ch(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(SarNet3ch, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
def sarnet1(pretrained=False, **kwargs):
    """Constructs SarNet model.
    
    Args: 
        pretrained (bool): If true, returns previously trained model
    """
    
    model = SarNet1ch(BasicBlock,[2,2,2,2], **kwargs)
    if pretrained:
        print("work in progress. will load trained model in future")
    return model

def sarnet2(pretrained=False, **kwargs):
    """Constructs SarNet model.
    
    Args: 
        pretrained (bool): If true, returns previously trained model
    """
    
    model = SarNet2ch(BasicBlock,[2,2,2,2], **kwargs)
    if pretrained:
        print("work in progress. will load trained model in future")
    return model


def sarnet3(pretrained=False, **kwargs):
    """Constructs SarNet model.
    
    Args: 
        pretrained (bool): If true, returns previously trained model
    """
    
    model = SarNet3ch(BasicBlock,[2,2,2,2], **kwargs)
    if pretrained:
        print("work in progress. will load trained model in future")
    return model

