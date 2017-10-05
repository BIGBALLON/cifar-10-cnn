# cifar-10-cnn


This repository is about some CNN Architecture's implementations for **cifar10**.  

![cifar10][1]

I just use **Keras** and **Tensorflow** to implementate all of these CNN models.

## Requirements

- Python (3.5.2)
- Keras (2.0.6)
- tensorflow-gpu (1.2.1)



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
| Vgg19-Network         |   0.5   |   meanstd  | GTX980TI  | 45M     |    4 hours    |    93.53    |
| Residual-Network50    |    -    |   meanstd  | GTX980TI  | 1.7M    |    8 h 58 min |    94.10    |
| Wide-resnet 16x8      |    -    |   meanstd  | GTX1060   | 11.3M   |  11 h 32 min  |    95.14    |
| DenseNet-100x12       |    -    |   meanstd  | GTX980TI  | 0.85M   |  30 h 40 min  |    95.15    |
| ResNeXt-4x64d         |    -    |   meanstd  | GTX1080TI | 20M     |  22 h 50 min  |    95.51    |
| SENet(ResNeXt-4x64d)  |    -    |   meanstd  | GTX1080   | 20M     |  -            |   -         |

## About ResNeXt & DenseNet

Because I don't have enough machines to train the larger networks.    
So I only trained the smallest network described in the paper.  
You can see the results in [liuzhuang13/DenseNet][11] and [prlz77/ResNeXt.pytorch][12]


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
  [11]: https://github.com/liuzhuang13/DenseNet
  [12]: https://github.com/prlz77/ResNeXt.pytorch