#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load Model

# In[ ]:


model = models.resnet152(pretrained=False)
num_ft = model.fc.in_features
model.fc = nn.Linear(num_ft, 200)
model_weight_filename="model_to_inference/best_0.7400000035762787_60.pkl"
model.load_state_dict(torch.load(model_weight_filename))
model = model.to(device)
model.eval()

set
# In[ ]:


test_transforms=transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.420, 0.490, 0.474],
         [0.263, 0.225, 0.230])])


# In[ ]:


with open("dataset/classes.txt", "r") as f:
    classes = f.read().split("\n")


# In[ ]:


with open('dataset/testing_img_order.txt') as f:
     test_images = [x.strip() for x in f.readlines()] 


# inference

# In[ ]:


submission = []


# In[ ]:


for i in range(len(test_images)):
    imagePath='dataset/testing_images/'+test_images[i]
    image = Image.open((imagePath)).convert('RGB')
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0).to(device)
    output = F.softmax(model(image_tensor)).data.cpu().numpy()[0]
    prediction = classes[np.argmax(output)]
    submission.append(prediction)


# In[ ]:


path='answer/answer.txt'
with open(path, 'w') as f:
    for i in range(len(test_images)):
        f.write(test_images[i]+' '+submission[i])
        f.write('\n')


# In[ ]:




