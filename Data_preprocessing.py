#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import shutil


# read class and label

# In[2]:


classes=[]
labels=[]


# In[3]:


with open ("dataset/classes.txt","r") as f:
    for line in f.readlines():
        line=line.strip('\n')
        classes.append(line)


# In[4]:


with open ("dataset/training_labels.txt","r") as f:
    for line in f.readlines():
        line=line.strip('\n')
        labels.append(line)


# In[5]:


for i in range(len(labels)):
    labels[i]=labels[i].split(' ')


# remove exist folder

# In[6]:


if os.path.exists('dataset/train_data'):
    shutil.rmtree('dataset/train_data')
    
if os.path.exists('dataset/val_data'):
    shutil.rmtree('dataset/val_data')
    
if os.path.exists('dataset/bird_dataset'):
    shutil.rmtree('dataset/bird_dataset')


# make folder for data

# In[7]:


os.mkdir('dataset/train_data')
os.mkdir('dataset/val_data')
os.mkdir('dataset/bird_dataset')


# In[8]:


for i in range(len(classes)):
    os.mkdir('dataset/val_data/'+classes[i])
    os.mkdir('dataset/train_data/'+classes[i])
    os.mkdir('dataset/bird_dataset/'+classes[i])


# sort data by class

# In[9]:


train_imgs=os.listdir('dataset/training_images')


# In[10]:


for i in range(len(train_imgs)):
    img=cv2.imread('dataset/training_images/'+train_imgs[i])
    for j in range(len(labels)):
        if (train_imgs[i]==labels[j][0]):
            cv2.imwrite('dataset/bird_dataset/'+labels[j][1]+'/'+train_imgs[i],img)


# split data.  train:val=8:2

# In[11]:


for i in range(len(classes)):
    data=os.listdir('dataset/bird_dataset/'+classes[i])
    for j in range(len(data)):
        img=cv2.imread('dataset/bird_dataset/'+classes[i]+'/'+data[j])
        if j<3:
            for k in range(len(labels)):
                if (data[j]==labels[k][0]):
                    cv2.imwrite('dataset/val_data/'+labels[k][1]+'/'+data[j],img)
        else:
            for k in range(len(labels)):
                if (data[j]==labels[k][0]):
                    cv2.imwrite('dataset/train_data/'+labels[k][1]+'/'+data[j],img)


# In[ ]:




