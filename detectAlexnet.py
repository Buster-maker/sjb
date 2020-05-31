import os
import time
# 全局取消证书验证

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import torch
from torch import nn
from torchvision import datasets, transforms

from Alexnet import AlexNet

batch_size=32

train_transforms = transforms.Compose([

        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_dir = './Atestdataset'
test_datasets = datasets.ImageFolder(test_dir, transform=train_transforms)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

model = AlexNet()
model.load_state_dict(torch.load('./Alex.pth',map_location=torch.device('cpu')))
for epoch in range(1):
    model = model.eval()
    total = 0
    correct = 0
    for i, data in enumerate(test_dataloader):
        images, labels = data
        vggoutputs = model(images)
        _, vggpredicted = torch.max(vggoutputs.data, 1)
        #print(labels)
        #print(vggpredicted)
        total += labels.size(0)
        correct += (vggpredicted == labels).sum().item()
    print(100.0 * correct / total)
