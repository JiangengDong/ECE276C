from robosuite.models.grippers import GRIPPER_MAPPING
from juggle_env.paddle_gripper import PaddleGripper
from juggle_env.juggle_env import JuggleEnv

GRIPPER_MAPPING["PaddleGripper"] = PaddleGripper
