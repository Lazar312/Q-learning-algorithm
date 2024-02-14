class Artifact:

    def __init__(self, pos=None):
        self.pos = pos

    @staticmethod
    def kind():
        pass

    @classmethod
    def image_path(cls):
        return f'img/{cls.kind()}.png'

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos


class Agent(Artifact):
    @staticmethod
    def kind():
        return 'a'


class Goal(Artifact):
    @staticmethod
    def kind():
        return 'x'
