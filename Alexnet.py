from torchvision import models
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=38):#num_classes=38
        super(AlexNet, self).__init__()
        net = models.alexnet(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
