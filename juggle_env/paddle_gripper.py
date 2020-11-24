import os

from robosuite.models.grippers.gripper_model import GripperModel


class PaddleGripper(GripperModel):
    """
    Gripper class to represent a ping-pong paddle

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/grippers/paddle_gripper.xml"))
        super().__init__(model_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 0

    @property
    def init_qpos(self):
        return None

    @property
    def _joints(self):
        return []

    @property
    def _actuators(self):
        return []

    @property
    def _contact_geoms(self):
        return []
