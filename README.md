# Time Series Explanation Spaces
This is the open source repository of "Explanation Space: A New Perspective into Time Series Interpretability" submitted to AAAI 2025.
The paper is available at: www.archix.org ...

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

