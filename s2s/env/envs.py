import numpy as np
import pygame

from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_drawer import _TreasureGameDrawer
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame as MTG
from gym_multi_treasure_game.envs.multiview_env import View
from s2s.env.s2s_env import S2SEnv
from s2s.image import Image


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

    def render_states(self, states: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image for the given states. This method can be overriden to optimise for the fact that there are
         multiple states. If not, it will simply average the results of render_state for each state
        """
        surface = None
        for state in states:
            if kwargs.get('randomly_sample', True):
                nan_mask = np.where(np.isnan(state))
                space = self.observation_space if kwargs.get('view', View.PROBLEM) == View.PROBLEM else self.agent_space
                state[nan_mask] = space.sample()[nan_mask]

            view = kwargs.get('view', View.PROBLEM)

            if view == View.PROBLEM:
                self._env.init_with_state(state)

            if self.drawer is None:
                self.drawer = _TreasureGameDrawer(self._env)

            if surface is None:
                surface = self.drawer.draw_to_surface()
            else:
                self.drawer.blend(surface, 0.5, 0.5)
        return pygame.surfarray.array3d(surface).swapaxes(0, 1)