from torch import nn
from torch.utils.data import DataLoader
from functools import reduce
import torch
from torchvision import models
class VGGNet(nn.Module):
    def __init__(self, num_classes=38):	   #num_classes，此处为 二分类值为2
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()	#将分类层置空，下面将改变我们的分类层
        self.features = net		  #保留VGG16的特征层
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(    #定义自己的分类层
                nn.Linear(512 * 7 * 7, 512),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
