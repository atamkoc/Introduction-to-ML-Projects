!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle

! kaggle datasets download misrakahmed/vegetable-image-dataset
! unzip vegetable-image-dataset

#all the imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms,models
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook
from IPython.core.display import HTML,display
from PIL import Image
from torch.utils.data import random_split
from torch.optim import SGD
import torch.optim as optim
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report


train_path=('/content/Vegetable Images/train')
validation_path=('/content/Vegetable Images/validation')
test_path=('/content/Vegetable Images/test')

classes = os.listdir(train_path)
classes

model=torchvision.models.vgg19(pretrained=True)

print(model)


# This is the implementation for our own VGG19Model

class VGG19(nn.Module):
    def __init__(self, my_pretrained_model, num_class=15):
        super(VGG19, self).__init__()
        self.pretrained = my_pretrained_model
        # modify last layer to match number of classes
        # Since the model itself has 6 layers and the number of outputs are related with the last layer we have to modify it according to our dataset
        in_nc = self.pretrained.classifier[6].in_features

        self.pretrained.classifier[6] = nn.Linear(in_nc, num_class)

    def forward(self, x):
        return self.pretrained(x)


#Creating a pretrained model in order to use with our VGG19 class implementation
pretrained_model = torchvision.models.vgg19(pretrained=True)

#Transforming part
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


#Creating datasets for train validation and test data
train_data=torchvision.datasets.ImageFolder(root=train_path,transform=transform)
val_data=torchvision.datasets.ImageFolder(root=validation_path,transform=transform)
test_data=torchvision.datasets.ImageFolder(root=test_path,transform=transform)

#getting sizes of datasets
train_size = len(train_data)
val_size = len(val_data)
test_size = len(test_data)

#checkin the sizes and making sure we are working with cuda instead of cpu in order to perform operations faster
print(train_size, val_size, test_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_vgg_model(model, optimizer, criterion, train_data_loader, val_data_loader, epoch_count):
    for epoch in range(epoch_count):  # loop over the dataset multiple times
        print("Epoch %d/%d #####################" % (epoch + 1, EPOCH_COUNT))

        # training part
        running_loss_train = 0.0
        running_true_train = 0

        model.train()

        for i, data in tqdm(enumerate(train_data_loader, 0), leave=False, total=len(train_data_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients

            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics

            running_loss_train += loss.item()
            running_true_train += torch.sum(pred == labels)

        train_loss = running_loss_train / train_size
        train_acc = running_true_train / train_size
        print("Training accuracy %f\tTraining average loss %f" % (train_acc, train_loss))

        ## VALIDATION PHASE ##
        running_loss_validation = 0.0
        running_true_validation = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validation_data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward + backward + optimize
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                # print statistics
                running_loss_validation += loss.item()
                running_true_validation += torch.sum(pred == labels)

        val_loss = running_loss_validation / val_size
        val_acc = running_true_validation / val_size
        print("Validation accuracy %f\Validation average loss %f" % (val_acc, val_loss))

    print('Finished Training')



#creating a model from our vgg19 class which is defined above

model = VGG19(pretrained_model)
for param in model.parameters():
    param.requires_grad = True
model.cuda()


optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
EPOCH_COUNT = 3
batch_size = 128

#creating loaders for train validation and test datasets
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#running training
train_vgg_model(model, optimizer, criterion, train_data_loader, validation_data_loader, EPOCH_COUNT)


y_pred = []
y_test = []
model.eval()
with torch.no_grad():
  for i, data in tqdm(enumerate(test_data_loader, 0), total=len(test_data_loader), leave=False):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward + backward + optimize
    outputs = model(inputs)
    pred = torch.argmax(outputs, dim=1)

    # print statistics
    y_pred.extend(pred.cpu().numpy())
    y_test.extend(labels.cpu().numpy())

from sklearn.metrics import classification_report
print(classification_report(y_pred, y_test, labels=np.arange(15)))

model_2 = VGG19(pretrained_model)
model_2.cuda()

## Omit all the layers
for param in model_2.parameters():
    param.requires_grad = False



##Then train the last two layers
for layer in model_2.pretrained.classifier[3:]:
    for param in layer.parameters():
        param.requires_grad = True


optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
EPOCH_COUNT = 3
batch_size = 128
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
train_vgg_model(model_2, optimizer, criterion, train_data_loader, validation_data_loader, EPOCH_COUNT)


y_pred = []
y_test = []
model_2.eval()
with torch.no_grad():
  for i, data in tqdm(enumerate(test_data_loader, 0), total=len(test_data_loader), leave=False):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward + backward + optimize
    outputs = model_2(inputs)
    pred = torch.argmax(outputs, dim=1)

    # print statistics
    y_pred.extend(pred.cpu().numpy())
    y_test.extend(labels.cpu().numpy())

print(classification_report(y_pred, y_test, labels=np.arange(15)))

#imports for visualization part
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as dataset

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


img=cv.imread('/content/Vegetable Images/test/Potato/1088.jpg')
#we have chosen the image above randomly to show how our layers affect it.

#plotting the image using matplotlib
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img=np.array(img)
img=transform(img)
img=img.unsqueeze(0)
print(img.size())

no_of_layers=0
conv_layers=[]

model.to(device)
model_children=list(model.children())

#getting convolutional layers
for x in model_children:
  for a in (x.features.to(device)):
    if type(a)==nn.Conv2d:
      no_of_layers+=1
      conv_layers.append(a.to(device))
    elif type(a)==nn.Sequential:
      for layer in a.children():
        if type(layer)==nn.Conv2d:
          no_of_layers+=1
          conv_layers.append(layer.to(device))


results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))
outputs = results

#applying each conv layer and plotting them using matplotlib
for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print("Layer ",num_layer+1)
    for i, filter in enumerate(layer_viz):
        if i == 16:
            break
        plt.subplot(2, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    plt.show()
    plt.close()

