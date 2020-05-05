import warnings

from typing import List

from gym_multi_treasure_game.envs.multiview_env import View
from linking_function import LinkingFunction
from s2s.core.learned_operator import LearnedOperator
from s2s.core.partitioned_option import PartitionedOption
from s2s.portable.quick_cluster import QuickCluster
from s2s.pddl.linked_operator import LinkedPDDLOperator
from s2s.pddl.pddl_operator import PDDLOperator

import numpy as np

from s2s.utils import show


class OperatorData:
    """
    This is just a general class for holding a partitioned option along with its associated learned operator and PDDL
    operators
    """

    def __init__(self, partitioned_option: PartitionedOption, learned_operator: LearnedOperator,
                 pddl_operators: List[PDDLOperator], **kwargs):
        """
        Associate a partitioned option with its operators
        :param partitioned_option: the partitioned option
        :param learned_operator: the learned precondition and effects
        :param pddl_operators: the PDDL operators
        """
        self._partitioned_option = partitioned_option
        # self._subpartitions = partitioned_option.subpartition(**kwargs)
        self._schemata = [LinkedPDDLOperator(x) for x in pddl_operators]
        self._learned_operator = learned_operator
        self.linking_function = LinkingFunction()
        # for _, states, _, next_states, _ in partitioned_option.effects(View.PROBLEM):
        #
        #     for s, s_prime in zip(states, next_states):
        #         i, _ = quick_cluster.get(s)
        #         j, _ = quick_cluster.get(s_prime)
        #         self.linking_function.add(i, j)

        # if len(self._subpartitions) == 0:
        #     #  Then we need to ground the precondition only
        #     view = View.PROBLEM if partitioned_option.view == View.PROBLEM else View.AGENT
        #     if view == View.PROBLEM:
        #         states = partitioned_option.problem_states
        #     else:
        #         states = partitioned_option.agent_states
        #     self._links = [Link(states, None, **kwargs)]
        # else:
        #     self._links = list()
        #     for subpartition in self._subpartitions:
        #         init_states = subpartition.states  # initial states based on subpartition's view
        #         for prob, _, _, next_states, _ in subpartition.effects():
        #             self._links.append(Link(init_states, next_states, probability=prob, **kwargs))
        #
        # self.linking_function = defaultdict(list)

    @property
    def schemata(self) -> List[LinkedPDDLOperator]:
        return self._schemata

    @property
    def links(self):
        return self.linking_function

    @property
    def option(self) -> int:
        return self._partitioned_option.option

    @property
    def partition(self) -> int:
        return self._partitioned_option.partition

    def add_link(self, quick_cluster: QuickCluster, state: np.ndarray, next_state: np.ndarray) -> None:
        i = quick_cluster.get(state, index_only=True)
        j = quick_cluster.get(next_state, index_only=True)
        self.linking_function.add(i, j)

    def link(self, quick_cluster: QuickCluster, verbose=False):

        used = set()
        for _, states, _, next_states, _ in self._partitioned_option.effects(View.PROBLEM):
            for s, s_prime in zip(states, next_states):
                self.add_link(quick_cluster, s, s_prime)
        for start, end, prob in self.links:
            if prob != 1:
                warnings.warn("Untested for case where linking prob != 1")
            used.add(start)
            show("Adding p_symbol{}".format(start), verbose)
            if end is None or start == end:
                end = -1
            else:
                used.add(end)
                show("Adding p_symbol{}".format(end), verbose)
            for operator in self._schemata:
                operator.add_link(start, end, prob)
        return used