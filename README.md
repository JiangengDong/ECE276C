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

    The initial values are [0.0, 0.7, 0.0, -1.4, 0.0, -0.56, 0.0]. The noises on the 7 joints are Gaussian and independent, with mu=0 and sigma=0.02.

1. Place the ball randomly in a initial region. 

    The position of the ball is uniformly distributed in the AABB [[0.76, -0.06, 1.96], [0.84, 0.04, 2.04]]. 

The observation is an ordered dict containing the following numpy arrays.

1. `"robot0_joint_pos"`
1. `"robot0_joint_vel"`
1. `"robot0_eef_pos"`
1. `"robot0_eef_quat"`
1. `"robot0_gripper_qpos"`
1. `"robot0_gripper_qvel"`
1. `"robot0_robot_state"`
1. `"pingpong_pos"`

### step

We use a joint velocity controller within the environment, so the input to the step function should be the velocity of each joints. The control frequency is 50Hz, and the maximum time span of an episode is 20s. The input should be in the range [-1, 1].

The reward consists of two parts: the state part and the control part. The state part is the distance on the x-y plane between the end-effector and ping-pong, added with a score that is achieved every time the ping-pong pass through the z=0.8 plane from downside to upside. The action part is to prevent the robot from taking rapid movement. 

### render & close

We provide two kinds of rendering: "human" and "rgb\_array". You can use `opencv` to generate a video by concatanating all the images returned by `env.render("rgb_array")`. One defect of our rendering functions is that it pops a window regardless the type you choose. 

## Training result

The results are put in the `data` folder. 
