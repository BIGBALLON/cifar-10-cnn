
## STEP 1: Install Ubuntu 16.04

---

1. 到 [Ubuntu官网][1] 下载ISO镜像, 刻录进U盘
2. 进入Win10，打开磁盘管理，压缩出足够的的磁盘空间(40GB or more)
3. Reboot，进入BIOS，关闭 Security boot 及 Win10 的 Fast boot (**Important**)
4. Reboot，从USB 引导进入安装界面
5. 选择Ubuntu与Windows共存，一路安装到底 

## STEP 2: Install NVIDIA Driver

---

- 更新源和必要的软件，如果在国内请自行更换合适的source

```
sudo apt-get update
sudo apt-get upgrade
```

- 禁用Nouveau

```
sudo vi /etc/modprobe.d/disable-nouveau.conf

//加入如下两行
blacklist nouveau
options nouveau modeset=0

```

- 重建kernel initramfs并重新启动

```
sudo update-initramfs -u
sudo reboot
```

- 安装NVIDIA 驱动

重启进入登录界面，切换到**tty1**(ctrl+alt+f1), 关闭lightdm图形界面

```
sudo service lightdm stop
```

增加 Nvidia 的 ppa 源

```
sudo apt-get purge nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
```

安装Nvidia

```
sudo apt-get install nvidia-375
reboot
```

重启，再次进入tty1，执行如下命令，没问题则ok了

```
sudo apt-get update && sudo apt-get upgrade
```

最后，用 nvidia-smi 查看GPU的信息

```
> nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.82                 Driver Version: 375.82                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 980 Ti  Off  | 0000:01:00.0      On |                  N/A |
|  0%   56C    P8    17W / 250W |    471MiB /  6076MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0       437    G   fcitx-qimpanel                                  11MiB |
|    0     32510    G   compiz                                         122MiB |
+-----------------------------------------------------------------------------+
```
## STEP 3: Install CUDA 8.0

---

- 从[官网][2]下载CUDA文件(**以cuda_8.0.61_375.26_linux.run为例**)
- 加执行权限并安装
- 安装时会询问是否安装显卡驱动，**务必选择No，前面已自行安装**

```
cd Downloads/
sudo chmod a+x cuda_8.0.61_375.26_linux.run
sudo ./cuda_8.0.61_375.26_linux.run
```

- 设置环境变量

```
sudo vi ~/.bashrc  
//加入这两行， LD_LIBRARY_PATH 和 CUDA_HOME 都不能少
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export CUDA_HOME=/usr/local/cuda-8.0
```

- 保存后，务必使环境变量生效

```
source ~/.bashrc
```

- 测试CUDA的Sample

```
cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery
sudo make
sudo ./deviceQuery
```

## STEP 4: Install Cudnn

---

- 到 [官网][3] 下载(需注册账号)， 解压 & 复制文件 & 加执行权限

```
cd Downloads/
tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz 
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*
```

## STEP 5: [Install Tensorflow][4] via pip

---

- 安装必要依赖

```
sudo apt-get install libcupti-dev
```
- 安装python-pip python-dev 并更新到最新版

```
sudo apt-get install python-pip python-dev
pip install -U pip
//or python3
sudo apt-get install python3-pip
pip3 install --upgrade pip
```

- 安装tensorflow

```
//python
sudo pip2 install tensorflow-gpu
//or python3
sudo pip3 install tensorflow-gpu
```

- 测试

开启一个terminal,这里我测试py3下的tensorflow。

```
bg@cgilab:~$ python3
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

输出，成功！

```
Hello, TensorFlow!
```

Take it easy!


  [1]: https://www.ubuntu.com/download/desktop
  [2]: https://developer.nvidia.com/cuda-downloads
  [3]: https://developer.nvidia.com/rdp/cudnn-download
  [4]: https://www.tensorflow.org/install/
  [5]: https://www.tensorflow.org/tutorials/mandelbrot
  [6]: http://7xi3e9.com1.z0.glb.clouddn.com/8899.png
