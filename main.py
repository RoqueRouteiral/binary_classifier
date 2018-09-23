import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from net import DenseNet
import torch.optim as optim
import time

def __one_epoch(loader, phase = 'train'):
    running_loss = 0.0
    correct = 0.0
    if phase == 'train':
        net.train()
    else:
        net.eval()
    for i, (image,target) in enumerate(loader):

        # forward + backward + optimize
        outputs = net(image)
        loss = criterion(outputs, target)
        
        # zero the parameter gradients
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        correct += (((outputs>0.5).byte()[:,1] == target.byte()).sum().item()/len(outputs))
        # print statistics
        
    return running_loss/len(loader), correct/len(loader)

list_of_images = os.listdir('fcbdata/')
#flags

train = False
test = True
epochs = 150
batch_size = 4
image_size = 96
trainset = 'fcbdata/train'
valset = 'fcbdata/validation'
testset = 'fcbdata/test'

#data loader
train_dataset = datasets.ImageFolder(
        trainset,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
#data loader
val_dataset = datasets.ImageFolder(
        valset,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()]))

test_dataset = datasets.ImageFolder(
        testset,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True)

#model definition and training parameters
net = DenseNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True, threshold=0.001)


if train:
    print('Starting training...')
    best_loss = np.inf
    losses = np.zeros(epochs)
    accs = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    val_accs = np.zeros(epochs)
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        losses[epoch], accs[epoch] = __one_epoch(train_loader,'train')
        val_losses[epoch], val_accs[epoch] = __one_epoch(val_loader,'val')
        
        end = time.time()
        print('[{}/{}] - Loss: {:.4f} Acc: {:.4f} Val Loss: {:.4f} Val Acc: {:.4f} Time: {:.4f} min'.format(epoch,epochs,losses[epoch],accs[epoch], val_losses[epoch], val_accs[epoch],(end - start)/60))
        if epoch >10 and losses[epoch]<best_loss:
            best_loss = val_losses[epoch]
            print('Saving model at epoch {}'.format(epoch))
            torch.save(net.state_dict(), 'model.ckpt')        
            
    print('Finished Training')
    torch.save(net.state_dict(), 'model.ckpt')
    
if test:
    print('Starting inference...')
    correct = 0.0
    net.load_state_dict(torch.load('model.ckpt'))
    start = time.time()
    net.eval()
    for i, (image,target) in enumerate(val_loader):
        # forward + backward + optimize
        outputs = net(image)>0.5
        # print statistics
        correct += ((outputs[:,1] == target.byte()).sum().item()/len(outputs))

    end = time.time()
    print('Acc: {} Time:{} min'.format(correct/len(val_loader),(end - start)/60))
    print('Finished testing')
    
