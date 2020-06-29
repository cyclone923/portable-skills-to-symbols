from collections import defaultdict

import PIL
from pygame.rect import Rect
from typing import List, Tuple, Dict

import numpy as np
import pygame

from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_drawer import _TreasureGameDrawer
from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_impl import create_options
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame as MTG
from gym_multi_treasure_game.envs.multiview_env import View
from s2s.env.s2s_env import S2SEnv
from s2s.image import Image
from s2s.portable.quick_cluster import QuickCluster


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

    @staticmethod
    def animate(version, plan: List[int], clusterer: QuickCluster) -> Dict[str, List[PIL.Image.Image]]:

        env = MultiTreasureGame(version_number=version)
        env = env._env

        pygame.init()
        # need the raw game!
        drawer = TreasureGameDrawer(env, clusterer, display_screen=True)
        clock = pygame.time.Clock()
        pygame.key.set_repeat()
        options, names = create_options(env, drawer=drawer)
        done = False
        count = 0
        while not done and count < 100:
            # execute until the plan works (may not work first time because stochasticity)
            env.reset_game()
            drawer.frames.clear()
            for option in plan:
                clock.tick(30)
                r = options[option]
                r.run()
                drawer.draw_domain()
                pygame.event.clear()
            done = env.player_got_goldcoin()  # got gold
            count += 1
            # done = env.player_got_goldcoin() and env.get_player_cell()[1] == 0  # got gold and returned

        pygame.display.quit()
        print("Extracting frames...")

        return {
            key: [PIL.Image.frombytes('RGBA', frame.get_size(), pygame.image.tostring(frame, 'RGBA', False)) for frame
                  in
                  frames] for key, frames in drawer.frames.items()
        }


class TreasureGameDrawer(_TreasureGameDrawer):

    def __init__(self, md, clusterer: QuickCluster, display_screen=False):
        super().__init__(md, display_screen=display_screen)
        self.frames = defaultdict(list)
        self.clusterer = clusterer
        pygame.font.init()

    def draw_domain(self, show_screen=True):
        super().draw_domain(show_screen=False)  # do not show it there, because we will do so below!
        if show_screen:
            myfont = pygame.font.SysFont('Courier', 28)
            text = 'Total actions: {0:04}'.format(self.env.total_actions)
            textsurface = myfont.render(text, False, (255, 255, 255))
            self.screen.blit(textsurface, (self.env.width - textsurface.get_size()[0] - 10, 10))
            pygame.display.flip()
            self.frames['normal'].append(self.screen.copy())
            self.frames['agent'].append(self.draw_local_view())
            self.frames['disambiguate'].append(self.draw_problem_specific())

    def draw_problem_specific(self):
        psymbol = self.clusterer.get(self.env.get_state())
        rough_state = psymbol.data
        x, y = rough_state[0], rough_state[1]
        x = int(max(0, (x - self.clusterer._threshold / 2) * self.env.width))
        y = int(max(0, (y - self.clusterer._threshold / 2) * self.env.height))
        width = int(self.clusterer._threshold * self.env.width)
        height = int(self.clusterer._threshold * self.env.height)

        s = pygame.Surface((width, height))  # the size of your rect
        s.set_alpha(128)
        s.fill((255, 0, 0))
        other = self.draw_background_to_surface()
        other.blit(s, (x, y))
        return other
