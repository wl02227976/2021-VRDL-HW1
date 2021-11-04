#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import config
from torch.optim import lr_scheduler


# parameter

# In[ ]:


train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        ##transforms.RandomAffine(0, shear=20),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.420, 0.490, 0.474],
                             [0.263, 0.225, 0.230])
        ])

test_valid_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.420, 0.490, 0.474],
                              [0.263, 0.225, 0.230])
        ])


# In[ ]:


train_directory = config.TRAIN_DATASET_DIR
valid_directory = config.VALID_DATASET_DIR

batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES


# In[ ]:


train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

valid_datasets = datasets.ImageFolder(valid_directory,transform=test_valid_transforms)
valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size)


# choose model

# In[ ]:


model = models.resnet50(pretrained=True) # Initialize the pretrained model
model = models.resnet152(pretrained=True)

num_ft = model.fc.in_features
model.fc = nn.Linear(num_ft, num_classes)  # replace final fully connected layer


loss_func = nn.CrossEntropyLoss()


# choose optimizer

# In[ ]:


##optimizer = optim.Adam(resnet50.parameters())
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)


# choose scheduler

# In[ ]:


##exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12800, gamma=0.1)


# In[ ]:


def train_and_valid(model, loss_function, optimizer, scheduler, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record = []
    best_acc = 0.0
    best_epoch = 0
    
    model = model.to(device)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()
        
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            
            optimizer.zero_grad()
            
            
            
            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)
            
            scheduler.step()

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                
                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        ##record_backup.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        

        if avg_valid_acc > best_acc  :
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), config.MODEL_SAVED_DIR+ 'best_'+str(best_acc) +'_'+ str(best_epoch + 1) + '.pkl')
            
            
            

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        
        ##torch.save(model.state_dict(), config.MODEL_SAVED_DIR+'bird_' + str(epoch + 1) + '.pkl')
    return model, record


# In[ ]:


if __name__=='__main__':
    num_epochs = config.NUM_EPOCHS
    ##record_backup=[]
    trained_model, record = train_and_valid(model, loss_func, optimizer,exp_lr_scheduler, num_epochs)
    ##torch.save(trained_model.state_dict(), config.TRAINED_MODEL)

    record = np.array(record)
    plt.plot(record[:, 0:2])
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.savefig(config.MODEL_SAVED_DIR+'loss.png')
    plt.show()

    plt.plot(record[:, 2:4])
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(config.MODEL_SAVED_DIR+'accuracy.png')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




