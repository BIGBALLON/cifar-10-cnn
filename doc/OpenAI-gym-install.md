# OpenAI Gym Install & Setting

Assume we already installed all of deep learning packages we need(tf, pytorch etc.)  


## Install openai-gym


In order to install openai-gym, we should setup the following packages:

```
sudo apt install cmake swig zlib1g-dev python3-tk -y
```

To save videos, we aslo need to install ffmpeg

```
sudo apt install ffmpeg
```

Then, we can setup openai-gym by the following commands:

```
git clone https://github.com/openai/gym
cd gym
pip3 install gym[all]
```

## Use fake screen to get videos


Because remote server has no display device, we can't see the training process and save videos.  
So we will use the following method to get the remote screen.


- Change GPU fan setting

```
sudo nvidia-xconfig -a --cool-bits=4
```


- Download the file [dfp-edid.bin][1] & move it to ``/etc/``

- Open xorg.conf

```
sudo vim /etc/X11/xorg.conf
```

- Insert the following three lines in each Section "Screen":

![screen][2]

```
Option      "UseDisplayDevice" "DFP-0"
Option      "ConnectedMonitor" "DFP-0"
Option      "CustomEDID" "DFP-0:/etc/dfp-edid.bin"

```

- Restart X server(ex. lightdm)

```
sudo service lightdm restart
```

- Auto login setting

bulid(or modify) file ``vim /etc/lightdm/lightdm.conf``  
write the following lines:

```
[Seat:*]
autologin-guest=false
autologin-user=username
autologin-user-timeout=0
```
then restart X server again.

- Install anydesk
    - Download & upload to your server(via sftp, scp or using wget etc.)
    - Install deb: ``sudo dpkg -i anydesk.XXX.deb``
    - Set password: ``anydesk --set-password``
    - Get ID: ``anydesk --get-id``
    - run anydesk ``anydesk``
    - (optinal) if it still doesn't work, try to use cmd ``export DISPLAY=:0``


## Use AnyDesk


- Download AnyDesk from [official website][3], windowsx64 version in this tutorial.  
- Double click to run the application(no need to install).
- Enter the ID we got from remote server.
- Enter the password.
- Then,you can see the remote desktop, enjoy it!

![anydesk][4]


## Run an simple code

```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
```

You can see the environment works fine!  
Take it easy!


![run][5]

## Reference


- [OpenAI Documentation][6]
- [Command Line Interface][7]


  [1]: https://1drv.ms/u/s!AiF_YAjgP2iqmUKHkH7t8lzlz9aO
  [2]: ./img/screen.png
  [3]: https://anydesk.com/platforms
  [4]: ./img/anydesk.png
  [5]: ./img/run.png
  [6]: https://gym.openai.com/docs/
  [7]: http://support.anydesk.com/knowledgebase/articles/441867-command-line-interface