# Solar Defect Detection using ResNet

This project implements a deep learning model using PyTorch to detect defects in solar cells. The model is based on the ResNet architecture and is designed to classify two types of defects: cracks and inactive regions.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [References](#references)

## Introduction
Solar modules are composed of many solar cells that are subject to degradation, causing different types of defects. This project focuses on detecting two types of defects:
1. **Cracks**: Ranging from very small to large cracks.
2. **Inactive regions**: Caused mainly by cracks, leading to disconnected parts of the cell that do not contribute to power production.

## Model Architecture
The model is based on the ResNet architecture. Below are the architectural details:

| Layer         | Description                                      |
|---------------|--------------------------------------------------|
| Conv2D        | Conv2D(3, 64, 7, 2)                              |
| BatchNorm     | BatchNorm()                                      |
| ReLU          | ReLU()                                           |
| MaxPool       | MaxPool(3, 2)                                    |
| ResBlock 1    | ResBlock(64, 64, 1)                              |
| ResBlock 2    | ResBlock(64, 128, 2)                             |
| ResBlock 3    | ResBlock(128, 256, 2)                            |
| ResBlock 4    | ResBlock(256, 512, 2)                            |
| GlobalAvgPool | AdaptiveAvgPool2d((1, 1))                        |
| Flatten       | Flatten()                                        |
| FC            | Linear(512, 2)                                   |
| Sigmoid       | Sigmoid()                                        |

Table 1: Architectural details for our ResNet. Convolutional layers are denoted by Conv2D(in_channels, out_channels, filter_size, stride). Max pooling is denoted MaxPool(pool_size, stride). ResBlock(in_channels, out_channels, stride) denotes one block within a residual network. Fully connected layers are represented by FC(in_features, out_features).

## Dataset
The dataset contains electroluminescence images of solar cells, provided in PNG format. The filenames and corresponding labels are listed in `data/data.csv`. Each row in the CSV file contains the path to an image and two numbers indicating if the solar cell shows a "crack" and if the solar cell can be considered "inactive".

## Setup
1. **Clone the repository**:
   ```sh
   git clone https://github.com/prashanthgadwala/SolarDefectDetection-ResNet.git
   cd SolarDefectDetection-ResNet

2. **Create and activate the conda environment**:
    ```sh
    conda env create -f environment.yml
    conda activate solar-defect-detection

3. **Install additional dependencies**
    ```sh
    pip install torch torchvision

4. **Extract the dataset**:
    ```sh
    unzip data/images.zip -d data/

5. **Run the train.py and main.py**:
    ```sh
    python train.py
    python main.py

5. **Hyperparameter Tuning**:
    Experiment with different hyperparameters to improve model performance. Common parameters to tune include learning rate and batch size. Observe and document how changes affect the performance.

    **Run all tests**:
        ```sh
        python3 PytorchChallengeTests.py
        python3 PytorchChallengeTests.py Bonus

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Electroluminescence Imaging](https://en.wikipedia.org/wiki/Electroluminescence)