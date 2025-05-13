# (new.ipynb)Custom Convolutional Neural Network for CIFAR-100 Image Classification
## Introduction 
Convolutional Neural Networks (CNNs) are widely used in the field of computer vision for
tasks such as image classification, object detection, and segmentation. The CIFAR-100
dataset, consisting of 60,000 images in 100 classes, presents a challenging task for deep
learning models. This assignment explores the design and implementation of a custom CNN
for image classification on the CIFAR-100 dataset. Additionally, the performance of the
custom CNN is compared with several well-established deep learning architectures: ResNet-
50, VGG-19, DenseNet-121, and EfficientNetB0. The goal is to understand the
key components of CNN architecture and evaluate their impact on model performance.
### Custom CNN Architecture
#### Architecture Design
The custom CNN architecture consists of three convolutional blocks followed by a fully
connected classification head. The following layers and techniques were used in the design:


<img src="arch.png">

The custom CNN architecture consists of three convolutional blocks, each followed by
batch normalization, ReLU activation, and max pooling. The first block uses 64 filters, the
second uses 128, and the third uses 256 filters, all with a kernel size of 3 ×3. These filter
sizes are standard in image classification tasks and help capture local spatial features at
increasing levels of abstraction. A kernel size of 3 ×3 strikes a balance between capturing
fine details and maintaining computational efficiency. Max pooling layers with a pool size of
2×2 are applied after each block to reduce spatial dimensions and retain dominant features.
