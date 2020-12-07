from robosuite.models import MujocoWorldBase
from robosuite.robots import SingleArm
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
from robosuite import load_controller_config

# typing
from robosuite.robots.robot import Robot
from robosuite.models.arenas import Arena
from robosuite.models.objects.generated_objects import MujocoGeneratedObject

from juggle_env.empty_arena import EmptyArena

from mujoco_py import MjSim, MjViewer
import gym
import numpy as np
from collections import OrderedDict
import glfw


class JuggleEnv:
    def __init__(self):
        self.control_freq: float = 50.0
        self.control_timestep: float = 1.0 / self.control_freq
        self.viewer = None
        self.horizon = 1000
        self.target = np.array([0.8, 0.0, 2.0])

        # load model
        self.robot: Robot = None
        self.arena: Arena = None
        self.pingpong: MujocoGeneratedObject = None
        self.model: MujocoWorldBase = None
        self._load_model()

        # initialize simulation
        self.mjpy_model = None
        self.sim: MjSim = None
        self.model_timestep: float = 0.0
        self._initialize_sim()

        # reset robot, object and internel variables
        self.cur_time: float = 0.0
        self.timestep: int = 0.0
        self.done: bool = False
        self._pingpong_body_id: int = -1
        self._paddle_body_id: int = -1
        self._reset_internel()

        # internel variable for scoring
        self._below_plane = False
        self.plane_height = 1.8

    def _load_model(self):
        # Load the desired controller's default config as a dict
        robot_noise = {
            "magnitude": [0.05]*7, 
            "type": "gaussian"
        }
        self.robot = SingleArm(
            robot_type="IIWA",
            idn=0,
            initial_qpos=[0.0, 0.7, 0.0, -1.4, 0.0, -0.7, 0.0],
            initialization_noise=robot_noise, 
            gripper_type="PaddleGripper",
            gripper_visualization=True,
            control_freq=self.control_freq
        )
        self.robot.load_model()
        self.robot.robot_model.set_base_xpos([0, 0, 0])

        self.arena = EmptyArena()
        self.arena.set_origin([0.8, 0, 0])

        self.pingpong = BallObject(
            name="pingpong",
            size=[0.04],
            rgba=[0.8, 0.8, 0, 1],
            solref=[0.1, 0.03],
            solimp=[0, 0, 1])
        pingpong_model = self.pingpong.get_collision()
        pingpong_model.append(new_joint(name="pingpong_free_joint", type="free"))
        pingpong_model.set("pos", "0.8 0 2.0")

        # merge into one
        self.model = MujocoWorldBase()
        self.model.merge(self.robot.robot_model)
        self.model.merge(self.arena)
        self.model.worldbody.append(pingpong_model)

    def _initialize_sim(self):
        # if we have an xml string, use that to create the sim. Otherwise, use the local model
        self.mjpy_model = self.model.get_model(mode="mujoco_py")

        # Create the simulation instance and run a single step to make sure changes have propagated through sim state
        self.sim = MjSim(self.mjpy_model)
        self.sim.step()
        self.robot.reset_sim(self.sim)
        self.model_timestep = self.sim.model.opt.timestep

    def _reset_internel(self):
        # reset robot
        self.robot.setup_references()
        self.robot.reset()

        # reset pingpong
        pingpong_pos = self.target + np.random.rand(3)*0.1-0.05
        pingpong_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos("pingpong_free_joint", np.concatenate([pingpong_pos, pingpong_quat]))

        # get handle for important parts
        self._pingpong_body_id = self.sim.model.body_name2id("pingpong")
        self._paddle_body_id = self.sim.model.body_name2id("gripper0_paddle_body")

        # Setup sim time based on control frequency
        self.cur_time = 0
        self.timestep = 0
        self.done = False

    def reset(self):
        self.sim.reset()
        self._reset_internel()
        self.sim.forward()
        return self._get_observation()

    def _get_observation(self):
        di = OrderedDict()

        # get robot observation
        di = self.robot.get_observations(di)

        # get pingpong observation
        pingpong_pos = np.array(self.sim.data.body_xpos[self._pingpong_body_id])
        di["pingpong_pos"] = pingpong_pos
        return di

    def step(self, action: np.ndarray):
        if self.done:
            raise ValueError("executing action in terminated episode")

        policy_step = True
        score = 0.0
        for _ in range(int(self.control_timestep / self.model_timestep)):
            self.sim.forward()
            self.robot.control(action=action, policy_step=policy_step)
            self.sim.step()
            policy_step = False
            # check if the ball pass the plane
            h = self.sim.data.body_xpos[self._pingpong_body_id][2]
            self._below_plane |= h < self.plane_height
            if self._below_plane and h > self.plane_height:
                score = 1.0
                self._below_plane = False

        self.timestep += 1
        self.cur_time += self.control_timestep
        self.done = self.timestep >= self.horizon
        observation = self._get_observation()
        print(observation["robot0_eef_pos"])
        reward = self._get_reward(observation, action) + score
        return observation, reward, self.done, {}

    def _get_reward(self, observation, action):
        diff = observation["robot0_eef_pos"] - observation["pingpong_pos"]
        return - 0.1*np.linalg.norm(diff[:2]) - 0.01*np.linalg.norm(action) 

    def render(self):
        self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            self.viewer.vopt.geomgroup[0] = 0
        return self.viewer

    def close(self):
        self._destroy_viewer()

    def _destroy_viewer(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def seed(self):
        pass
