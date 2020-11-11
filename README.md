# ECE276C

This project, which is a final project of the ECE276C course in UCSD, aims to train a robot to juggle a table tennis ball. 

## Dependencies' installation guide

I assume the system is Ubuntu 18.04 and the Python version is 3.7.9 in this part. 

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

1. Add the following line to the end of `~/.bashrc`. 

    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
    ```

### mujoco-py

1. Install some packages with `apt-get`.

    ```bash
    sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    ```

1. Install `mujoco-py` with `pip`. You can also install it with [the other packages](#other-python-packages).

### Other Python packages

I prepare a `requirements.txt` that specifies all the necessary Python packages with their versions fixed. You can install with the following command. 

```bash
pip install -r requirements.txt
```