import gym


class MaxLength(gym.Wrapper):
    """
    A wrapper that limits the number of option executions in an episode before terminating
    """

    def __init__(self, env: gym.Env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, agent_obs, reward, done, info = super().step(action)  # TODO FIX: Assuming we're getting agent view
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, agent_obs, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ConstantLength(MaxLength):
    """
    A wrapper that specifies the exact number of options executions that can be executed in an episode. The environment
    MUST support the ability to continue execution even when its own done flag has been set.
    """

    def step(self, action):
        observation, agent_obs, reward, done, info = super().step(action)  # TODO FIX: Assuming we're getting agent view
        if self._elapsed_steps != self._max_episode_steps:
            info['force_continue'] = True  # if it's not exactly this, keep going!!
        return observation, agent_obs, reward, done, info


class ConditionalAction(gym.Wrapper):
    """
    Wrapper to deal with the case of actions not being valid. The environment must provide a function called can_execute
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        can_execute = self.env.can_execute(action)
        observation, agent_obs, reward, done, info = super().step(action)  # TODO FIX: Assuming we're getting agent view
        info['option_failed'] = not can_execute
        return observation, agent_obs, reward, done, info


class ActionExecutable(gym.Wrapper):
    """
    Wrapper that adds information about which actions are available at the current and next state. Each is given as a
    binary vector
    """

    def step(self, action):
        current_actions = self.env.available_mask
        observation, agent_obs, reward, done, info = super().step(action)  # TODO FIX: Assuming we're getting agent view
        info['current_actions'] = current_actions
        info['next_actions'] = self.env.available_mask
        return observation, agent_obs, reward, done, info
