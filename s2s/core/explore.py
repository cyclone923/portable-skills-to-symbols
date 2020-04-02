import random

import multiprocessing
import numpy as np
import pandas as pd
from functools import partial
from gym import Wrapper

from s2s.core.data.multiviewframe import TransitionFrame
from s2s.env.envs import MultiTreasureGame
from s2s.env.s2s_env import S2SWrapper
from s2s.utils import show, run_parallel, save

__author__ = 'Steve James and George Konidaris'


def collect_data(env: S2SWrapper, max_timestep=np.inf, max_episode=np.inf, verbose=False, seed=None, n_jobs=1,
                 **kwargs) -> (
        pd.DataFrame, pd.DataFrame):
    """
    Collect data from the environment through uniform random exploration in parallel

    :param env: the environment
    :param max_timestep: the maximum number of timesteps in total (not to be confused with maximum time steps per episode) Default is infinity
    :param max_episode: the maximum number of episodes. Default is infinity
    :param verbose: whether to print additional information
    :param seed: the random seed. Use for reproducibility
    :param n_jobs: the number of processes to spawn to collect data in parallel. If -1, use all CPUs
    :return: data frames holding transition and initation data
    """
    if max_timestep == np.inf and max_episode == np.inf:
        raise ValueError('Must specify at least a maximum timestep or episode limit')

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # run collection in parallel
    max_timestep /= n_jobs
    max_episode /= n_jobs

    functions = [
        partial(_collect_data, env, np.random.randint(0, 1000000), max_timestep, max_episode, verbose,
                int(max_episode * i), **kwargs)
        for i in range(n_jobs)]

    results = run_parallel(functions)
    transition_data = pd.concat([x[0] for x in results], ignore_index=True)
    initiation_data = pd.concat([x[1] for x in results], ignore_index=True)
    return transition_data, initiation_data


def _collect_data(env: S2SWrapper, seed=None, max_timestep=np.inf, max_episode=np.inf, verbose=False,
                  episode_offset=0, **kwargs) -> (
        pd.DataFrame, pd.DataFrame):
    """
    Collect data from the environment through uniform random exploration
    :param env: the environment
    :param seed: the random seed. Use for reproducibility
    :param max_timestep: the maximum number of timesteps in total (not to be confused with maximum time steps per episode) Default is infinity
    :param max_episode: the maximum number of episodes. Default is infinity
    :param verbose: whether to print additional information
    :return: data frames holding transition and initation data
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    transition_data = pd.DataFrame(
        columns=['episode', 'state', 'agent_state', 'option', 'reward', 'next_state', 'next_agent_state', 'done',
                 'goal_achieved', 'mask', 'agent_mask', 'next_options'])
    initiation_data = pd.DataFrame(columns=['state', 'agent_state', 'option', 'can_execute'])

    n_episode = 0
    n_timesteps = 0
    while n_episode < max_episode and n_timesteps < max_timestep:
        show('Running episode {}'.format(n_episode + episode_offset), verbose)
        state, agent_state = env.reset()
        done = False
        ep_timestep = 0
        while not done and n_timesteps < max_timestep:
            action = env.sample_action()
            next_state, next_agent_state, reward, done, info = env.step(action)
            failed = info.get('option_failed', False)
            # timestep only counts if we actually executed an option
            if not failed:
                n_timesteps += 1
                # mask = np.where(np.array(state) != np.array(next_state))[0]  # check which indices are not equal!
                mask = np.where(np.array(state) != np.array(next_state))[0]
                agent_mask = np.where(np.array(agent_state) != np.array(next_agent_state))[0]
                next_options = info.get('next_actions', np.array([]))
                success = info.get('goal_achieved', False)
                transition_data.loc[len(transition_data)] = [n_episode + episode_offset, state, agent_state, action,
                                                             reward, next_state, next_agent_state, done, success, mask,
                                                             agent_mask, next_options]
                ep_timestep += 1
            if 'current_actions' in info:
                # the set of options that could or could not be executed
                for i, label in enumerate(info['current_actions']):
                    initiation_data.loc[len(initiation_data)] = [state, agent_state, i, bool(label)]
            else:
                # just use the information we have
                initiation_data.loc[len(initiation_data)] = [state, agent_state, action, not failed]
            show('\tStep: {}'.format(ep_timestep), verbose and ep_timestep > 0 and ep_timestep % 50 == 0)
            state = next_state
            agent_state = next_agent_state
        n_episode += 1
    return transition_data, initiation_data
