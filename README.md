
# 2021 VRDL HW1

This repository is the official implementation of [2021 VRDL HW1](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07). 


## Reproducing Submission
1. [Requirements](#Requirements)
2. [Pretrained_Model](#Pretrained_Model)(After downloading, put it into 'model_to_inference' folder.)
3. [Dataset](#Dataset)(Download the testing_images,unzip it and put it into 'dataset' folder)
4. [Inference](#Inference)

## Retrain
1. [Requirements](#Requirements)
2. [Dataset](#Dataset)
3. [Config](#Config)
4. [Training](#Training)
5. [Config](#Config) (the model to inference='your model name')
6. [Inference](#Inference)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Config
Set the parameters
```parameter
number of classes
batch_size
number of epochs
the path to save model
dataset folder
the model to inference
```


## Dataset 
Download the dataset from [data](https://drive.google.com/drive/folders/1G1cZ8BE4oJf469zLKordpghwx9Mmg4k8?usp=sharing)
After downloading, put them into dataset folder.

```data
python Data_preprocessing.py
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py
```
After training, choose a highest model from result folder and put it into model_to_inference folder.


## Inference

To inference model, run:

```eval
python inference.py
```


## Pretrained_Model

You can download pretrained model here:

- [best_0.7400000035762787_60.pkl](https://drive.google.com/file/d/1nUxSO_0VJfWdwXmgqa54iXVetuxREJ8B/view?usp=sharing)

After downloading, put it into 'model_to_inference' folder.

## Results

Our model achieves the following performance on :


| Model name                   | local_acc        | codalab_acc    |
| ------------------           |----------------  | -------------- |
| best_0.7400000035762787_60   |     0.74         |   0.678866     |



## Reference
https://pse.is/3np5cg

https://github.com/wangzhebufangqi/MachineLearning
