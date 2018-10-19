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


PS: try to change learning rate setting and retrain the network to see different results.
