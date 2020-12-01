# ECE276C

This project, which is a final project of the ECE276C course in UCSD, aims to train a robot to juggle a table tennis ball. 

## Dependencies' installation guide

We assume the system is Ubuntu 18.04 and the Python version is 3.7.9 in this part. 

### MuJoCo

1. Get a trial license or a students license from <https://www.roboti.us/license.html>. An email with a `mjkey.txt` attachment will be sent to your mailbox.

1. Download [MuJoCo 200](https://www.roboti.us/download/mujoco200_linux.zip) and unzip it to the `~/.mujoco/` folder. After unzipping, the folder hierarchy should be similar to the following one. 

    ```
    ~/.mujoco
    └── mujoco200
        ├── bin
        ├── doc
        ├── include
        ├── model
        └── sample
    ```

1. Copy the `mjkey.txt` to two folders: `~/.mujoco/` and `~/.mujoco/mujoco200/bin`. After copying, the folder hierarchy should be as follows. 

    ```
    ~/.mujoco
    ├── mjkey.txt
    └── mujoco200
        ├── bin
        │   ├── mjkey.txt
        │   └── ...
        ├── doc
        ├── include
        ├── model
        └── sample
    ```

1. Append the following line to the end of `~/.bashrc`. 

    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
    ```

### mujoco-py

1. Install some packages with `apt-get`.

    ```bash
    sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libglew-dev
    ```

1. Append the following line to the end of `~/.bashrc`. 

    ```bash
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    ```

1. Install `mujoco-py` with `pip`. You can also install it with [the other packages](#other-python-packages).

### Other Python packages

I prepare a `requirements.txt` that specifies all the necessary Python packages with their versions fixed. You can install with the following command. 

```bash
pip install -r requirements.txt
```

### Verify installation

If everything is installed properly, you can test the `JuggleEnv` with the following demo code. 

```python
from juggle_env import JuggleEnv
import numpy as np 
import time

env = JuggleEnv()
ob = env.reset()
for _ in range(500):
    env.step(np.zeros((7, )))
    env.render()
    time.sleep(0.02)
env.close()
```

## Environment design

The `JuggleEnv` is composed of two parts: an 7-dof IIWA robot and a ping-pong ball. We adopt the gym's API, and the details of the methods are explained below.

### reset 

During reset, two operation are taken. 

1. Set the manipulator's joints back to the initial values with some noise. 

    The initial values are [0.0, 0.7, 0.0, -1.4, 0.0, -0.7, 0.0]. The noises on the 7 joints are Gaussian and independent, with mu=0 and sigma=0.05.

1. Place the ball randomly in a initial region. 

    The position of the ball is uniformly distributed in the AABB [[0.75, -0.05, 1.95], [0.85, 0.05, 2.05]].

### reward

The reward consists of two parts: state part and control part. The state part is to enforce the ball to bounce between the paddle and a max height. The action part is to prevent the robot from taking rapid movement. 

### step

We use a joint velocity controller within the environment, so the input to the step function should be the velocity of each joints. The control frequency is 50Hz, and the maximum time span of an episode is 20s. 

### render & close

Till now, we only provide a "human" rendering. If you want to record a video, press "v" key during the execution. This will produce a `video-00000.mp4` under the `/tmp/` directory. 