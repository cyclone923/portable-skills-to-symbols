from gym.spaces import Discrete, Box

from gym_treasure_game.envs._treasure_game_impl._treasure_game_impl import _TreasureGameImpl, create_options
from s2s.env.s2s_env import S2SEnv
from gym_treasure_game.envs import TreasureGame as TG
from s2s.utils import make_path, get_dir_name
import numpy as np

class TreasureGame(TG, S2SEnv):

    def __init__(self):
        super().__init__()


class TreasureGameVX(TreasureGame):

    def __init__(self, version_number: int):
        self._version_number = version_number
        dir = make_path(get_dir_name(__file__), 'layouts')
        self._env = _TreasureGameImpl(make_path(dir, 'domain_v{}.txt'.format(version_number)),
                                      make_path(dir, 'domain-objects_v{}.txt'.format(version_number)),
                                      make_path(dir, 'domain-interactions_v{}.txt'.format(version_number)))
        self.drawer = None
        self.option_list, self.option_names = create_options(self._env)
        self.action_space = Discrete(len(self.option_list))
        s = self._env.get_state()
        self.observation_space = Box(np.float32(0.0), np.float32(1.0), shape=(len(s),))
        self.viewer = None

    def __str__(self):
        return "TreasureGameV{}".format(self._version_number)

    def describe_option(self, option: int) -> str:
        return self.option_names[option]

