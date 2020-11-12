from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import NullGripper
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

from mujoco_py import MjSim, MjViewer

import numpy as np
import torch
from matplotlib import pyplot as plt
import tensorboard
import tianshou
from tqdm import tqdm
import gym


def test_robosuite():
    world = MujocoWorldBase()

    mujoco_robot = Panda()
    mujoco_robot.add_gripper(NullGripper())
    mujoco_robot.set_base_xpos([0, 0, 0])
    world.merge(mujoco_robot)

    mujoco_arena = TableArena()
    mujoco_arena.set_origin([0.8, 0, 0])
    world.merge(mujoco_arena)

    sphere = BallObject(
        name="sphere",
        size=[0.04],
        rgba=[0, 0.5, 0.5, 1]).get_collision()
    sphere.append(new_joint(name='sphere_free_joint', type='free'))
    sphere.set('pos', '1.0 0 1.0')
    world.worldbody.append(sphere)

    model = world.get_model(mode="mujoco_py")
    return model


def test_mujoco(model):
    sim = MjSim(model)
    viewer = MjViewer(sim)
    viewer.vopt.geomgroup[0] = 0  # disable visualization of collision mesh

    for _ in range(10000):
        sim.data.ctrl[:] = 0
        sim.step()
        viewer.render()


def main():
    model = test_robosuite()
    test_mujoco(model)


if __name__ == "__main__":
    main()
