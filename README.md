# Convolutional Neural Networks for CIFAR-10 


This repository is about some implementations of CNN Architecture  for **cifar10**.  

![cifar10][1]

I just use **Keras** and **Tensorflow** to implementate all of these CNN models.  
(maybe torch/pytorch version if I have time)

## Requirements

- Python (3.5)
- keras (>= 2.1.5)
- tensorflow-gpu (>= 1.4.1)

## Architectures and papers

- The first CNN model: **LeNet**    
    - [LeNet-5 - Yann LeCun][2]
- **Network in Network**
    - [Network In Network][3]
- **Vgg19 Network**
    -  [Very Deep Convolutional Networks for Large-Scale Image Recognition][4]
    -  The **1st places** in ILSVRC 2014 localization tasks
    -  The **2nd places** in ILSVRC 2014 classification tasks 
- **Residual Network** 
    -  [Deep Residual Learning for Image Recognition][5]
    -  [Identity Mappings in Deep Residual Networks][6]
    -  **CVPR 2016 Best Paper Award**
    -  **1st places** in all five main tracks:
        - ILSVRC 2015 Classification: "Ultra-deep" 152-layer nets
        - ILSVRC 2015 Detection: 16% better than 2nd
        - ILSVRC 2015 Localization: 27% better than 2nd
        - COCO Detection: 11% better than 2nd
        - COCO Segmentation: 12% better than 2nd
-  **Wide Residual Network**
    -  [Wide Residual Networks][7]
-  **ResNeXt**  
    -  [Aggregated Residual Transformations for Deep Neural Networks][8]
    -  Used in [Mask-RCNN][9]
-  **DenseNet**
    -  [Densely Connected Convolutional Networks][10]
    -  **CVPR 2017 Best Paper Award**
-  **SENet**
    - [Squeeze-and-Excitation Networks][11]  
    - **The 1st places** in ILSVRC 2017 classification tasks 

## Documents & tutorials

There are also some documents and tutorials in [doc][12] & [issues/3][13].  
Get it if you need.   
You can aslo see the [articles][14] if you can speak Chinese. <img src="https://user-images.githubusercontent.com/7837172/44953504-b9481000-aec8-11e8-9920-abf66365b8d8.gif">



## Accuracy of all my implementations

**In particular**：  
Change the batch size according to your GPU's memory.  
Modify the learning rate schedule may imporve the results of accuracy!  

| network               | GPU       | params  | batch size | epoch | training time | accuracy(%) |
|:----------------------|:---------:|:-------:|:----------:|:-----:|:-------------:|:-----------:|
| Lecun-Network         | GTX1080TI | 62k     |   128      |  200  |    30 min     |    76.23    |
| Network-in-Network    | GTX1080TI | 0.97M   |   128      |  200  |    1 h 40 min |    91.63    |
| Vgg19-Network         | GTX1080TI | 39M     |   128      |  200  |    1 h 53 min |    93.53    |
| Residual-Network20    | GTX1080TI | 0.27M   |   128      |  200  |    44 min     |    91.82    |
| Residual-Network32    | GTX1080TI | 0.47M   |   128      |  200  |    1 h 7 min  |    92.68    |
| Residual-Network50    | GTX1080TI | 1.7M    |   128      |  200  |    1 h 42 min |    93.18    |
| Residual-Network110   | GTX1080TI | 0.27M   |   128      |  200  |    3 h 38 min |    93.93    |
| Wide-resnet 16x8      | GTX1080TI | 11.3M   |   128      |  200  |   4 h 55 min  |    95.13    |
| Wide-resnet 28x10     | GTX1080TI | 36.5M   |   128      |  200  |   10 h 22 min |    95.78    |
| DenseNet-100x12       | GTX1080TI | 0.85M   |   64       |  250  |   17 h 20 min |    94.91    |
| DenseNet-100x24       | GTX1080TI | 3.3M    |   64       |  250  |   22 h 27 min |    95.30    |
| DenseNet-160x24       | 1080 x 2  | 7.5M    |   64       |  250  |   50 h 20 min |    95.90    |
| ResNeXt-4x64d         | GTX1080TI | 20M     |   120      |  250  |   21 h 3 min  |    95.19    |
| SENet(ResNeXt-4x64d)  | GTX1080TI | 20M     |   120      |  250  |   21 h 57 min |    95.60    |


## About LeNet and CNN training tips/tricks

LeNet is the first CNN network proposed by LeCun.   
I used different CNN training tricks to show you how to train your model efficiently.  

``LeNet_keras.py`` is the baseline of LeNet,  
``LeNet_dp_keras.py`` used the Data Prepossessing [DP],  
``LeNet_dp_da_keras.py`` used both DP and the Data Augmentation[DA],  
``LeNet_dp_da_wd_keras.py`` used DP, DA and Weight Decay [WD]  

| network                |  GPU      | DP      |     DA     |   WD  | training time | accuracy(%) |
|:-----------------------|:---------:|:-------:|:----------:|:-----:|:-------------:|:-----------:|
| LeNet_keras            | GTX1080TI |   -     |   -        |   -   |    5 min      |    58.48    |
| LeNet_dp_keras         | GTX1080TI |   √     |   -        |   -   |    5 min      |    60.41    |
| LeNet_dp_da_keras      | GTX1080TI |   √     |   √        |   -   |    26 min     |    75.06    |
| LeNet_dp_da_wd_keras   | GTX1080TI |   √     |   √        |   √   |    26 min     |    76.23    |

For more CNN training tricks, see [Must Know Tips/Tricks in Deep Neural Networks][15] (by [Xiu-Shen Wei][16])

## About Learning Rate schedule

**Different learning rate schedule** may get **different training/testing accuracy!**  
See **[./htd][17]**, and **[HTD][18]** for more details.  

## About [Multiple GPUs Training][19] 

Since the latest version of Keras is already supported ``keras.utils.multi_gpu_model``, so you can simply use the following code to train your model with multiple GPUs:

```python
from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50

model = ResNet50()

# Replicates `model` on 8 GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',optimizer='adam')

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```


## About ResNeXt & DenseNet

Since I don't have enough machines to train the larger networks, I only trained the smallest network described in the paper.  You can see the results in [liuzhuang13/DenseNet][20] and [prlz77/ResNeXt.pytorch][21]

<a href="https://bigballon.github.io">&nbsp;<img src="https://user-images.githubusercontent.com/7837172/44953504-b9481000-aec8-11e8-9920-abf66365b8d8.gif"></a> <a href="https://bigballon.github.io">&nbsp;<img src="https://user-images.githubusercontent.com/7837172/44953504-b9481000-aec8-11e8-9920-abf66365b8d8.gif"></a>

Please feel free to contact me if you have any questions! 


  [1]: ./images/cf10.png
  [2]: http://yann.lecun.com/exdb/lenet/
  [3]: https://arxiv.org/abs/1312.4400
  [4]: https://arxiv.org/abs/1409.1556
  [5]: https://arxiv.org/abs/1512.03385
  [6]: https://arxiv.org/abs/1603.05027
  [7]: https://arxiv.org/abs/1605.07146
  [8]: https://arxiv.org/abs/1611.05431
  [9]: https://arxiv.org/abs/1703.06870
  [10]: https://arxiv.org/abs/1608.06993
  [11]: https://arxiv.org/abs/1709.01507
  [12]: ./doc
  [13]: https://github.com/BIGBALLON/cifar-10-cnn/issues/3
  [14]: https://zhuanlan.zhihu.com/dlgirls
  [15]: http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
  [16]: http://lamda.nju.edu.cn/weixs/
  [17]: ./htd
  [18]: https://github.com/BIGBALLON/HTD
  [19]: https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus
  [20]: https://github.com/liuzhuang13/DenseNet
  [21]: https://github.com/prlz77/ResNeXt.pytorch
