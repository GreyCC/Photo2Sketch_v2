# Photo2SketchV2 (Quality Photo Sketch with Improved Deep Learning Structure)

By Chun-Chun Hui, Wan-Chi Siu, Wen-Xin Zhang and H. Anthony Chan

This repo provides simple testing codes, pretrained models and the demonstration executive file for simple testing.

We propose a photo to sketch convolutional neural network that output quality photo sketch with simple improved structure on our network structure

# BibTex

To be added

# Implementation

## Training

Download MS COCO 2017 dataset by https://cocodataset.org/#download

You will need 2017 Train images [118K/18GB], 2017 Train/Val annotations [241MB]

Unzip 2017 Train images into data/coco/train2017/

Unzip 2017 Train/Val annotations into data/coco/edge/

## Testing

Copy your image to folder "Test" and run

```py
python eval_test.py
```
