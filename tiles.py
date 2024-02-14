import inspect
import sys

from artifacts import Artifact


class Tile(Artifact):
    def reward(self):
        pass


class Grass(Tile):
    def reward(self):
        return -5

    @staticmethod
    def kind():
        return 'g'


class Mud(Tile):
    def reward(self):
        return -50

    @staticmethod
    def kind():
        return 'm'


class Hole(Tile):
    def reward(self):
        return -1000

    @staticmethod
    def kind():
        return 'h'


class TilesFactory:
    tiles_map = {obj.kind(): obj
                 for name, obj in inspect.getmembers(sys.modules['tiles'])
                 if inspect.isclass(obj) and name not in {'Artifact', 'Tile'}}

    @staticmethod
    def generate_tile(kind, pos=None):
        return TilesFactory.tiles_map[kind](pos)
