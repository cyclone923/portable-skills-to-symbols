import numpy as np
import pygame

from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_drawer import _TreasureGameDrawer
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame as MTG
from gym_multi_treasure_game.envs.multiview_env import View
from s2s.env.s2s_env import S2SEnv


class MultiTreasureGame(MTG, S2SEnv):

    def _render_state(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image of the given state. There should be no missing state variables (using render_state if so)
        """

        view = kwargs.get('view', View.PROBLEM)

        if view == View.PROBLEM:
            self._env.init_with_state(state)
            return self.render(mode='rgb_array', view=view)
        elif view == View.AGENT:
            if self.drawer is None:
                self.drawer = _TreasureGameDrawer(self._env)
            # draw the agent view too
            surface = self.drawer.draw_local_view(state)
            return pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame

        raise ValueError
