from typing import List

import pandas as pd

from gym_multi_treasure_game.envs.multiview_env import View
from s2s.utils import pd2np

__author__ = 'Steve James and George Konidaris'


class PartitionedOption:
    """
    This class holds the data representing a single partitioned option
    """

    def __init__(self, option: int, partition: int, combined_data: pd.DataFrame, effects: List[pd.DataFrame],
                 view: View, look_similar=None):
        """
        Create a new partitioned option
        :param option: the option index
        :param partition: the partition index
        :param combined_data: all the data (including probabilistic effects) concatenated
        :param effects: the individiaul effects (each effect is a stochastic transition)
        :param view: the view
        :param look_similar: a set of other partition indices that look identical to this one (but are in fact not)
        """
        if look_similar is None:
            look_similar = set()
        self._option = option
        self._view = view
        self._partition = partition
        self._states = pd2np(combined_data['state'])
        self._agent_states = pd2np(combined_data['agent_state'])

        state_column = 'next_state' if view == View.PROBLEM else 'next_agent_state'
        mask_column = 'mask' if view == View.PROBLEM else 'agent_mask'

        total_samples = sum(len(effect[state_column]) for effect in effects)
        self._effects = [(len(effect[state_column]) / total_samples, effect[['state', 'agent_state', 'reward',
                                                                             'next_state', 'next_agent_state',
                                                                             mask_column]])
                         for effect in effects]
        self._look_similar = look_similar  # other partitions that look similar but are not
        self._combined_data = combined_data

    def is_similar(self, other_partition: int):
        """
        Determines whether the given partition looks similar in agent space, but is in fact not. In this case, it should
        not be included as negative precondition samples, because they will be identical to the current partition's
        precondition samples!
        :param other_partition: the index of the other partition being compared
        """
        return other_partition in self._look_similar or other_partition == self.partition

    @property
    def option(self):
        return self._option

    @property
    def partition(self):
        return self._partition

    @property
    def view(self):
        return self._view

    @property
    def states(self):
        if self._view == View.PROBLEM:
            return self.problem_states
        elif self._view == View.AGENT:
            return self.agent_states

    @property
    def problem_states(self):
        return self._states

    @property
    def agent_states(self):
        return self._agent_states

    def effects(self, view=None):

        if view is None:
            view = self._view

        state_modifier = '' if view == View.PROBLEM else 'agent_'
        mask_modifier = '' if self._view == View.PROBLEM else 'agent_'

        for probability, frame in self._effects:
            yield probability, pd2np(frame['{}state'.format(state_modifier)]), pd2np(frame['reward']), pd2np(
                frame['next_{}state'.format(state_modifier)]), pd2np(frame['{}mask'.format(mask_modifier)]).astype(int)

    def subpartition(self, verbose=False, **kwargs) -> List['PartitionedOption']:
        """
        Given the current partition, partition it again in the alternate space (that is, if it was partitioned in
        agent space initially, now partition in problem space)
        :param verbose: the verbosity level
        :return: (sub) partitioned options in the opposite space
        """
        other_view = View.PROBLEM if self._view == View.AGENT else View.AGENT  # swap the view around
        from s2s.core.partition import _partition_option
        return _partition_option(self.option, self._combined_data, verbose=verbose, view=other_view, **kwargs)

