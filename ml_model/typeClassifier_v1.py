import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from resnet import *

import matplotlib.pyplot as plt
import numpy as np

import os

def make_label_list(root):
    dir_list = os.listdir(root)
    classes = list()
    for i, dir in enumerate(dir_list, 0):
        if i != 0:
            #print(dir)
            classes.append(dir)

    return classes
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = make_label_list(
    root="/media/disk1/STUFF/eddible_ones/data/datasets/dataset-test/")
print("We have {} classes".format(len(classes)))

transformations = torchvision.transforms.Compose([
    # you can add other transformations in this list
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(
    root="/media/disk1/STUFF/eddible_ones/data/datasets/dataset",
    transform=transformations)

test_dataset = torchvision.datasets.ImageFolder(
    root="/media/disk1/STUFF/eddible_ones/data/datasets/dataset-test/",
    transform=transformations)
#print(train_dataset)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(6 * 110 * 110, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        x = x.view(-1, 6 * 110 * 110)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)
'''
net = resnet50().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #print(data)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Loss: {}'.format(running_loss / (i+1)))

print('Finished Training')

PATH = '/media/disk1/STUFF/eddible_ones/model_library/resnet50/model.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

net = resnet50()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
#print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
