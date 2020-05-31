import os
import time
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from model import VGGNet
batch_size=1
lr=0.0001
save_dir='./save_weight_files/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def save_network(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.state_dict(), save_path)

# train_transforms = transforms.Compose([
#
#         # transforms.RandomHorizontalFlip(p=0.5),
#         # transforms.RandomRotation(30, resample=False, expand=False, center=None),
#         # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#         transforms.Scale(227),
#         transforms.Resize(224),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
train_dir = './Atraindataset'
test_dir = './Atestdataset'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_datasets = datasets.ImageFolder(test_dir,transform=train_transforms)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)
# .cuda()
model = VGGNet().to(device)
lossF = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.999))
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
def Accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total
for epoch in range(200):
    model = model.train()
    Total_accurcy = []
    for i, data in enumerate(train_dataloader):
        input,label=data
        # input, label =input.to(device),label.to(device)
        output = model(input)
        optimizer.zero_grad()
        loss = lossF(output, label)
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            pred_y = torch.max(output, 1)[1]
            accuracy = float((pred_y == label).sum()) / float(label.size(0))
            Total_accurcy.append(accuracy)
            Accurcy = sum(Total_accurcy) / len(Total_accurcy)
            print("Epoch:%d/%d Batch: %d/%d  ----- loss:%f-- trian accuracy: %.4f" % (
                    epoch, 200, i, len(train_dataloader),loss, accuracy))#--test accuracy: %.4f
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        save_network(model,'Vgg16',str(epoch))
        torch.save(model.state_dict(), './VGG16.pth')