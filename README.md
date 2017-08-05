# cifar-10-cnn
Using cifar-10 datasets to learn deep learning.

```
 cifar-10-cnn
 |__ 1_Lecun_Network
    |__LeNet_dp_da_wd_keras.py
 |__ 2_Network_in_Network
    |__Network_in_Network_keras.py
 |__ 3_Vgg19_Network
    |__Vgg19_keras.py
 |__ 4_Residual_Network
    |__ResNet_keras.py
 |__ 5_Wide_Residual_Network
    |__Wide_ResNet_keras.py
 |__ 6_ResNeXt (TODO)
 |__ 7_DenseNet (TODO)
 |__ Tensorflow_version
     |__data_utility.py
     |__Network_in_Network.py
     |__Network_in_Network_bn.py
     |__vgg_19.py
     |__vgg_19_pretrain.py
     |__(TODO)
```


| network            | dropout | preprocess | GPU       | epochs  | training time | accuracy(%) |
|:------------------:|:-------:|:----------:|:---------:|:-------:|:-------------:|:-----------:|
| Lecun-Network      |    -    |   meanstd  | GTX980TI  | 180     |    30 min     |    76.27    |
| Network-in-Network |   0.5   |   meanstd  | GTX1060   | 164     |    1 h 30 min |    91.15    |
| Vgg19-Network      |   0.5   |   meanstd  | GTX980TI  | 164     |    4 hours    |    93.43    |
| Residual-Network50 |   0.5   |   meanstd  | GTX980TI  | 200     |    8 h 58 min |    94.10    |
| Wide-resnet 16x8   |   0.5   |   meanstd  | GTX1060   | 200     |  11 h 32 min  |    95.14    |



