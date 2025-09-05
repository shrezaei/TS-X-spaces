# Time Series Explanation Spaces
This is the open source repository for [Explanation Space: A New Perspective into Time Series Interpretability](https://arxiv.org/abs/2409.01354) and [On the Necessity of Multi-Domain Explanation: An Uncertainty Principle Approach for Deep Time Series Models](https://arxiv.org/pdf/2506.03267?) both published in ICDM 2025.

## Library Versions
* Python 3.10.9
* Pytorch 2.2.1
* Captum 0.7.0
* tsai 0.3.9
* tslearn 0.6.3

## Train a Target Model
Using the following command, you can trained a ResNet model on GunPoint dataset. You can choose any dataset from UCR repository available in tslearn package.
```
$ python train.py -m ResNet -d GunPoint
```
To generate an explanation with DeepLIFT, use the following command
```
$ python Explain.py -m ResNet -d GunPoint -a DeepLift -x Time
```
You can change the explanation space with -x option. Current options include 'Time', 'Freq', 'TimeFreq', 'Diff', 'MinZero', and 'Diff_back_to_Time'.
```
$ python Explain.py -m ResNet -d GunPoint -a DeepLift -x Freq
```
To check if the uncertainty principle is violated you can set the space variable as 'Uncertainty_principle_test'.
```
$ python Explain.py -m ResNet -d GunPoint -a DeepLift -x Uncertainty_principle_test
```
