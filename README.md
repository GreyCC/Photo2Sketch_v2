# Quality Photo Sketch with Improved Deep Learning Structure

Link to this paper: https://ieeexplore.ieee.org/abstract/document/9978022

Chun-Chun Hui, Wan-Chi Siu, Wen-Xin Zhang and H. Anthony Chan, "Quality Photo Sketch with Improved Deep Learning Structure", Proceedings, IEEE Region 10 Conference (TENCON 2022), 01-04 November 2022, Hong Kong

Download paper directly: [Quality Photo Sketch with Improved Deep Learning Structure.pdf](https://github.com/GreyCC/Photo2Sketch_v2/files/10039634/Quality.Photo.Sketch.with.Improved.Deep.Learning.Structure.pdf)

This repo provides simple testing codes, pretrained models and the demonstration executive file for simple testing.

We propose a photo to sketch convolutional neural network that output quality photo sketch with simple improved structure on our network structure

# BibTex

```
@INPROCEEDINGS{9978022,
  author={Hui, Chun-Chuen and Siu, Wan-Chi and Zhang, Wen-Xin and Chan, H. Anthony},  
  booktitle={TENCON 2022 - 2022 IEEE Region 10 Conference (TENCON)},   
  title={Quality Photo Sketch with Improved Deep Learning Structure},   
  year={2022}, 
  volume={},  
  number={},  
  pages={1-6},  
  doi={10.1109/TENCON55691.2022.9978022}}
```

# Implementation

## Training

Download MS COCO 2017 dataset by https://cocodataset.org/#download

You will need ***2017 Train images [118K/18GB]***

Unzip Train images into data/coco/train2017/

Unzip Segmentation images into data/coco/edge/

Run

```py
python main.py
```

## Testing (Code base)

Copy your test images to folder Test/example/ and run

```py
python eval_test.py
```
Results are generated at Test/our/

## Testing (.exe) [Window only]

Make sure you also download folder 'model', 'model' and .exe file are under same directory 

Open **sketch.exe**

Select test image by clicking button 'Select image'

'Photo2Sketch' will process sketch image by our previous method

'Photo2Sketch_v2' will process sketch image by our improved method

Click 'Save image' to save result in your computer
