# cifar-10-cnn


This repository is about some CNN Architecture's implementations for **cifar10**.  
I just use **Keras** and **Tensorflow** to implementate all of these CNN models.

## Requirements

- Python (3.5.2)
- Keras (2.0.6)
- tensorflow-gpu (1.2.1)


## Architectures and papers

- The first CNN model: **LeNet**    
    - [LeNet-5 - Yann LeCun][1]
- **Network in Network**
    - [Network In Network][2]
- **Vgg19 Network**
    -  [Very Deep Convolutional Networks for Large-Scale Image Recognition][3]
- **Residual Network**
    -  [Deep Residual Learning for Image Recognition][4]
    -  [Identity Mappings in Deep Residual Networks][5]
-  **Wide Residual Network**
    -  [Wide Residual Networks][6]
-  **ResNeXt**(TODO)
    -  [Aggregated Residual Transformations for Deep Neural Networks][7]
-  **DenseNet**(TODO)
    -  [Densely Connected Convolutional Networks][8]

## Accuracy of all my implementations

| network            | dropout | preprocess | GPU       | epochs  | training time | accuracy(%) |
|:------------------:|:-------:|:----------:|:---------:|:-------:|:-------------:|:-----------:|
| Lecun-Network      |    -    |   meanstd  | GTX980TI  | 180     |    30 min     |    76.27    |
| Network-in-Network |   0.5   |   meanstd  | GTX1060   | 164     |    1 h 30 min |    91.15    |
| Vgg19-Network      |   0.5   |   meanstd  | GTX980TI  | 164     |    4 hours    |    93.43    |
| Residual-Network50 |   0.5   |   meanstd  | GTX980TI  | 200     |    8 h 58 min |    94.10    |
| Wide-resnet 16x8   |   0.5   |   meanstd  | GTX1060   | 200     |  11 h 32 min  |    95.14    |


  [1]: http://yann.lecun.com/exdb/lenet/
  [2]: https://arxiv.org/abs/1312.4400
  [3]: https://arxiv.org/abs/1409.1556
  [4]: https://arxiv.org/abs/1512.03385
  [5]: https://arxiv.org/abs/1603.05027
  [6]: https://arxiv.org/abs/1605.07146
  [7]: https://arxiv.org/abs/1611.05431
  [8]: https://arxiv.org/abs/1608.06993
