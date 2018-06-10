Result of ResNet-50 (we run it 5 times and show “best” as in the following table)  
see [HTD](https://arxiv.org/abs/1806.01593) for more details if you are interested.

![image](https://user-images.githubusercontent.com/7837172/41199099-92ab4058-6cbe-11e8-9846-7046e747e538.png)


| Network               | Methods    | runs            | **CIFAR-10**  | **CIFAR-100** |
|:----------------------|:----------:|:---------------:|:-------------:|:-------------:|
| ResNet-50             |step decay  |  med. of 5 runs | 93.09%        | 70.25%        |
| ResNet-50             |cos         |  med. of 5 runs | 93.42%        | 70.07%        |
| ResNet-50             |HTD         |  med. of 5 runs | **93.58%**    | **70.81%**    |
