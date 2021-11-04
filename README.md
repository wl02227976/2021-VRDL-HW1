
# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Reproducing Submission
1. [Requirements](#Requirements)
2. 
3. [Pretrained Model](#Pretrained Model)
4. [Inference](#Inference)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
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


## Inference

To evaluate my model on ImageNet, run:

```eval
python inference.py
```


## Pretrained Model

You can download pretrained model here:

- [best_0.7400000035762787_60.pkl](https://drive.google.com/file/d/1nUxSO_0VJfWdwXmgqa54iXVetuxREJ8B/view?usp=sharing)

After downloading, put it into model_to_inference folder

## Results

Our model achieves the following performance on :


| Model name                   | local_acc        | codalab_acc    |
| ------------------           |----------------  | -------------- |
| best_0.7400000035762787_60   |     0.74         |   0.678866     |



## Reference
https://pse.is/3np5cg

https://github.com/wangzhebufangqi/MachineLearning
