import gym
import numpy as np
from abc import ABC, abstractmethod

from gym_multi_treasure_game.envs.multiview_env import MultiViewEnv, View
from s2s.image import Image
from s2s.wrappers import ConditionalAction, ConstantLength, ActionExecutable


class S2SWrapper(gym.Wrapper):

    def __init__(self, env: 'S2SEnv', options_per_episode=np.inf):

        if not isinstance(env.unwrapped, MultiViewEnv):
            raise ValueError("Environment must inherit from MultiViewEnv")

        env = ConditionalAction(env)  # support actions that are not executable everywhere
        env = ConstantLength(env, options_per_episode)  # restrict episode lengths
        env = ActionExecutable(env)  # determine at each state which actions/options are executable
        super().__init__(env)

    def step(self, action):
        state, observation, reward, done, info = super().step(action)
        if done and not info.get('TimeLimit.truncated', False):
            info['goal_achieved'] = True
        if info.get('force_continue', False):
            done = False  # force the domain to continue on

        return state, observation, reward, done, info


class S2SEnv(gym.Env, ABC):
    """
    An environment that forces the user to implement the neccesary methods
    """

    @property
    @abstractmethod
    def available_mask(self) -> np.ndarray:
        """
        Return a binary array specifying which options can be run at the current state
        :return:
        """
        pass

    def can_execute(self, action):
        return self.available_mask[action] == 1

    def sample_action(self, valid_only=True):
        """
        Randomly pick an action
        :param valid_only: whether only valid actions should be picked
        :return: an action
        """
        if not valid_only:
            return self.action_space.sample()
        mask = self.available_mask
        return np.random.choice(np.arange(self.action_space.n), p=mask / mask.sum())

    def render_states(self, states: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image for the given states. This method can be overriden to optimise for the fact that there are
         multiple states. If not, it will simply average the results of render_state for each state
        """
        return Image.merge([self.render_state(state, **kwargs) for state in states])

    def render_state(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image of the given state. Where state variables are missing, specify with np.nan
        """
        if kwargs.get('randomly_sample', True):
            nan_mask = np.where(np.isnan(state))
            space = self.observation_space if kwargs.get('view', View.PROBLEM) == View.PROBLEM else self.agent_space
            state[nan_mask] = space.sample()[nan_mask]
        return self._render_state(state, **kwargs)

    def _render_state(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image of the given state. There should be no missing state variables (using render_state if so)
        """
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    def describe_option(self, option: int) -> str:
        return 'option-{}'.format(option)