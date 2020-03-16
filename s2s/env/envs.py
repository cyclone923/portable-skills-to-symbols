import gym
import pygame
from gym.envs.classic_control import rendering
from gym.spaces import Discrete, Box

from gym_treasure_game.envs._treasure_game_impl._treasure_game_drawer import _TreasureGameDrawer
from gym_treasure_game.envs._treasure_game_impl._treasure_game_impl import _TreasureGameImpl, create_options
from s2s.env.s2s_env import S2SEnv, MultiViewEnv
from gym_treasure_game.envs import TreasureGame as TG
from s2s.image import Image
from s2s.utils import make_path, get_dir_name
import numpy as np


class TreasureGame(TG, S2SEnv):

    def __init__(self):
        super().__init__()

    def _render_state(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image of the given state. There should be no missing state variables (using render_state if so)
        """
        self._env.init_with_state(state)
        return self.render(mode='rgb_array')


class TreasureGameVX(TreasureGame, MultiViewEnv):

    @property
    def agent_space(self) -> gym.Space:
        return Box(low=np.float32(0), high=np.float32(1), shape=(11,))

    def current_agent_observation(self) -> np.ndarray:
        return self._env.current_observation()

    def __init__(self, version_number: int):

        if version_number == 0:
            # use original
            super().__init__()
            return

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

    def _render_state(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image of the given state. There should be no missing state variables (using render_state if so)
        """
        self._env.init_with_state(state)
        return self.render(mode='rgb_array')

    def render(self, mode='human', view='problem'):
        if self.drawer is None:
            self.drawer = _TreasureGameDrawer(self._env)

        self.drawer.draw_domain()
        local_rgb = None
        if view == 'agent':
            # draw the agent view too
            surface = self.drawer.draw_local_view()
            local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame

        rgb = pygame.surfarray.array3d(self.drawer.screen).swapaxes(0, 1)  # swap because pygame
        if mode == 'rgb_array':
            return local_rgb if view == 'agent' else rgb
        elif mode == 'human':
            # draw it like gym
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            if view == 'agent':
                a = Image.to_image(rgb, mode='RGB')
                b = Image.to_image(local_rgb, mode='RGB')
                rgb = Image.to_array(Image.combine([a, b]))
            self.viewer.imshow(rgb)
