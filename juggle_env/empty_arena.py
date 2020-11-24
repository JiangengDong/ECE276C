import os

from robosuite.models.arenas import Arena


class EmptyArena(Arena):
    """Empty workspace."""

    def __init__(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/arenas/empty_arena.xml"))
        super().__init__(model_path)
        self.floor = self.worldbody.find("./geom[@name='floor']")