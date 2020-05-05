import time
from collections import deque
from typing import List
import matplotlib.pyplot as plt

import gym
import numpy as np
from gym.envs.classic_control import rendering

from gym_multi_treasure_game.envs.multiview_env import View
from s2s.core.learned_operator import LearnedOperator
from s2s.core.link_operators import find_closest_start_partition
from s2s.env.envs import MultiTreasureGame
from s2s.env.s2s_env import S2SEnv
from s2s.portable.operator_data import OperatorData
from s2s.portable.problem_symbols import ProblemSymbols
from s2s.utils import load


def show(next_states):
    im = env.render_states(next_states, view=View.AGENT, randomly_sample=False)
    # viewer.imshow(im.astype(int))
    plt.imshow(im.astype(int))
    plt.show()


def evaluate_plan(env: S2SEnv, operators: List[OperatorData], problem_symbols: ProblemSymbols, plan: List[int],
                  use_rewards=True, n_samples=100, verbose=False):
    """
    Evaluate a plan
    :param env: the domain
    :param operators: the learned operators
    :param plan: the plan
    :param use_rewards: whether to measure based on reward collected, or probability of success
    :param n_samples: the number of samples to use as the empirical state distribution
    :param verbose: the verbosity level
    :return: the probability of executing the plan, or the expected return along the plan
    """

    class _Node:
        """
        Node for a DFS/BFS search
        """

        def __init__(self, states: np.ndarray, current_partition: int, remaining_plan: List[int], prob=1, reward=0,
                     parent: '_Node' = None):
            self.states = states
            self.plan = remaining_plan
            self.prob = prob
            self.reward = reward
            self.parent = parent
            self.current_partition = current_partition

        @property
        def is_leaf(self):
            return len(self.plan) == 0

        @property
        def neighbours(self):
            if self.is_leaf:
                return []
            neighbours = list()
            option = self.plan[0]
            candidates = [x for x in operators if x.option == option]
            for candidate in candidates:

                if self.current_partition not in candidate.linking_function:
                    continue

                learned_operator = candidate._learned_operator

                pre_prob = learned_operator.precondition.probability(self.states)
                if pre_prob < 0.05:
                    continue
                for partition_prob, next_partition in candidate.linking_function[self.current_partition]:
                    for next_prob, eff, reward in learned_operator.outcomes():
                        next_states = np.copy(self.states)
                        next_states[:, eff.mask] = np.around(eff.sample(next_states.shape[0]))

                        prob = self.prob * pre_prob * next_prob * partition_prob
                        if reward is None:
                            rew = 0
                        else:
                            rew = self.reward + pre_prob * next_states * np.mean(
                                [reward.predict_reward(x) for x in self.states])
                        neighbours.append(
                            _Node(next_states, next_partition, self.plan[1:], prob=prob, reward=rew, parent=current))

            return sorted(neighbours, key=lambda x: x.prob, reverse=True)

    state, obs = env.reset()

    start_partition = None
    distance = np.inf

    for proposition in problem_symbols:
        mean = np.mean(proposition.sample(100), axis=0)
        if np.linalg.norm(mean - state, np.inf) < distance:
            distance = np.linalg.norm(mean - state, np.inf)
            start_partition = proposition

    temp = str(start_partition)
    start_partition = int(temp[temp.index('_') + 1: -1])

    # obs = np.array([1,1,1,2,9,0,1,1,1,0,0])

    state_dist = np.vstack([obs for _ in range(n_samples)])


    stack = deque()
    current = _Node(state_dist, start_partition, plan)
    stack.append(current)
    leaves = list()
    while len(stack) > 0:
        current = stack.pop()
        if current.is_leaf:
            leaves.append(current)
        for next in current.neighbours:
            stack.append(next)

    max_prob = 0
    max_leaf = None
    for leaf in leaves:
        if leaf.prob > max_prob:
            max_prob = leaf.prob
            max_leaf = leaf

    if max_leaf is not None:
        path = deque()
        x = max_leaf
        while x is not None:
            path.append(x)
            x = x.parent
        while len(path) > 0:
            x = path.pop()
            im = env.render_states(x.states, view=View.AGENT, randomly_sample=False)
            plt.imshow(im.astype(int))
            plt.show()
            plt.pause(0.5)

    return max_prob


if __name__ == '__main__':
    env = MultiTreasureGame(version_number=1)
    operators = load('output/linked_operator_data.pkl')
    problem_symbols = load('output/problem_symbols.pkl')

    plan = [3, 1, 3, 6, 1, 8, 2, 1, 2, 0, 4, 0, 1, 1, 3, 4, 1, 0, 0]
    plan = [3]
    # plan = [0, 1, 1, 3, 4, 1, 0, 0]
    prob = evaluate_plan(env, operators, problem_symbols, plan)
    print(prob)
    #                    0       1         2           3            4        5            6          7          8
    # option_list = [go_left, go_right, up_ladder, down_ladder, interact, down_left, down_right, jump_left, jump_right]
