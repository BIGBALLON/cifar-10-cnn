# Convolutional Neural Networks (CNN) for CIFAR-10 Dataset


This repository is about some CNN Architecture's implementations for **cifar10**.  

![cifar10][1]

I just use **Keras** and **Tensorflow** to implementate all of these CNN models.  
(maybe pytorch version if I have time)

## Requirements

- Python (3.5.2)
- Keras (2.0.8)
- tensorflow-gpu (1.3.0)


## Architectures and papers

- The first CNN model: **LeNet**    
    - [LeNet-5 - Yann LeCun][2]
- **Network in Network**
    - [Network In Network][3]
- **Vgg19 Network**
    -  [Very Deep Convolutional Networks for Large-Scale Image Recognition][4]
- **Residual Network**
    -  [Deep Residual Learning for Image Recognition][5]
    -  [Identity Mappings in Deep Residual Networks][6]
-  **Wide Residual Network**
    -  [Wide Residual Networks][7]
-  **ResNeXt**  
    -  [Aggregated Residual Transformations for Deep Neural Networks][8]
-  **DenseNet**
    -  [Densely Connected Convolutional Networks][9]
-  **SENet**
    - [Squeeze-and-Excitation Networks][10]  


## Accuracy of all my implementations

| network               | dropout | preprocess | GPU       | params  | training time | accuracy(%) |
|:----------------------|:-------:|:----------:|:---------:|:-------:|:-------------:|:-----------:|
| Lecun-Network         |    -    |   meanstd  | GTX980TI  | 62k     |    30 min     |    76.27    |
| Network-in-Network    |   0.5   |   meanstd  | GTX1060   | 0.96M   |    1 h 30 min |    91.25    |
| Network-in-Network_bn |   0.5   |   meanstd  | GTX980TI  | 0.97M   |    2 h 20 min |    91.75    |
| Vgg19-Network         |   0.5   |   meanstd  | GTX980TI  | 39M     |    4 hours    |    93.53    |
| Residual-Network110   |    -    |   meanstd  | GTX980TI  | 1.7M    |    8 h 58 min |    94.10    |
| Wide-resnet 16x8      |    -    |   meanstd  | GTX1060   | 11.3M   |  11 h 32 min  |    95.14    |
| DenseNet-100x12       |    -    |   meanstd  | GTX980TI  | 0.85M   |  30 h 40 min  |    95.15    |
| ResNeXt-4x64d         |    -    |   meanstd  | GTX1080TI | 20M     |  22 h 50 min  |    95.51    |
| SENet(ResNeXt-4x64d)  |    -    |   meanstd  | GTX1080   | 20M     |  -            |   -         |

Now, I fixed some bugs and used 1080TI to retrain all of the following models.  

**In particular**：  
Change the batch size according to your GPU's memory.  
Modify the learning rate schedule may imporve the results of accuracy!  

| network               | GPU       | params  | batch size | epoch | training time | accuracy(%) |
|:----------------------|:---------:|:-------:|:----------:|:-----:|:-------------:|:-----------:|
| Lecun-Network         | GTX1080TI | 62k     |   128      |  200  |    30 min     |    76.25    |
| Network-in-Network    | GTX1080TI | 0.97M   |   128      |  200  |    1 h 40 min |    91.63    |
| Vgg19-Network         | GTX1080TI | 39M     |   128      |  200  |    1 h 53 min |    93.53    |
| Residual-Network20    | GTX1080TI | 0.27M   |   128      |  200  |    47 min     |    92.16    |
| Residual-Network32    | GTX1080TI | 0.47M   |   128      |  200  |    1 h 13 min |    92.86    |
| Residual-Network110   | GTX1080TI | 1.7M    |   128      |  200  |    4 h 30 min |    94.44    |
| Wide-resnet 16x8      | GTX1080TI | 11.3M   |   128      |  200  |   5 h 1 min   |    95.13    |
| DenseNet-100x12       | GTX1080TI | 0.85M   |   64       |  250  |   17 h 20 min |    94.91    |
| DenseNet-100x24       | GTX1080TI | 3.3M    |   64       |  250  |   22 h 27 min |    95.30    |
| DenseNet-160x24       | 1080 x 2  | 7.5M    |   64       |  250  |   50 h 20 min |    95.90    |
| ResNeXt-4x64d         | GTX1080TI | 20M     |   120      |  250  |   21 h 3 min  |    95.19    |
| SENet(ResNeXt-4x64d)  | GTX1080TI | 20M     |   120      |  250  |   21 h 57 min |    95.60    |


## About ResNeXt & DenseNet

Because I don't have enough machines to train the larger networks.    
So I only trained the smallest network described in the paper.  
You can see the results in [liuzhuang13/DenseNet][12] and [prlz77/ResNeXt.pytorch][13]


  [1]: ./images/cf10.png
  [2]: http://yann.lecun.com/exdb/lenet/
  [3]: https://arxiv.org/abs/1312.4400
  [4]: https://arxiv.org/abs/1409.1556
  [5]: https://arxiv.org/abs/1512.03385
  [6]: https://arxiv.org/abs/1603.05027
  [7]: https://arxiv.org/abs/1605.07146
  [8]: https://arxiv.org/abs/1611.05431
  [9]: https://arxiv.org/abs/1608.06993
  [10]: https://arxiv.org/abs/1709.01507
  [11]: ./images/results.jpg
  [12]: https://github.com/liuzhuang13/DenseNet
  [13]: https://github.com/prlz77/ResNeXt.pytorch