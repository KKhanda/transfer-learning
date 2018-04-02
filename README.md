# Transfer Learning in Keras
## Model description
Model used: Inception V3

Layers added: MaxPooling, Dropout, Convolution, Flattening and Full connection
Epochs: 3
Batch size: 64
Steps per epoch: 10
Validation: 10 validation steps after each of three epochs
 
Accuracy on test set (without additional Convolution2D layer): ~83%
After I changed MaxPooling to shrink layers in 2 times instead of 8 times 
and applied Convolution2D, the accuracy increased as following
Accuracy on test set: ~95%

## Dataset description
Images of cats and dogs

## Installation
```bash
$ git clone
$ pip3 install requirements.txt
```
## Launch
```bash
$ python3 transfer-learning.py
```
