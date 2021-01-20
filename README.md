# Paddle2.0 implementation of GSM

PaddlePaddle implementation for the video recognition module GSM.

["Gate-Shift Networks for Video Action Recognition, CVPR, 2020." By Swathikiran Sudhakaran, Sergio Escalera, Oswald Lanz](http://openaccess.thecvf.com/content_CVPR_2020/papers/Sudhakaran_Gate-Shift_Networks_for_Video_Action_Recognition_CVPR_2020_paper.pdf)

[The author's Repository for GSM](https://github.com/swathikirans/GSM)


## Introduction

This repository contains all the required codes and results for the [Paddle2.0](https://github.com/paddlepaddle) implementation of the paper Gate-Shift Networks for Video Action Recognition. 

## Environment

Python 3.7.4

PaddlePaddle 2.0.0-rc1

## Repository Structure

* model: the GSM model with InceptionV3 as the backbone
* result: the training information saved as npz, accuray vs. epochs
* dataset.py: for dataset declaration
* datasets_video.py: for dataset reading
* extract_frames_diving48.py: for extracting frames from videos
* process_dataset_diving.py: for generating training and testing list files
* train.py: for training
* test.py: for testing
* transforms.py: for data augmentation and normalization
* requirements.txt: the required packages
* README.md: repository description file
