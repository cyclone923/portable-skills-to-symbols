from collections import defaultdict

from typing import List

from gym_multi_treasure_game.envs.multiview_env import View
from s2s.core.learned_operator import LearnedOperator
from s2s.core.partitioned_option import PartitionedOption
from s2s.pddl.linked_operator import LinkedPDDLOperator
from s2s.pddl.pddl_operator import PDDLOperator

import numpy as np

from s2s.pddl.pddl import Proposition
from s2s.portable.link import Link
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
        self._subpartitions = partitioned_option.subpartition(**kwargs)
        self._schemata = [LinkedPDDLOperator(x) for x in pddl_operators]
        self._learned_operator = learned_operator

        if len(self._subpartitions) == 0:
            #  Then we need to ground the precondition only
            view = View.PROBLEM if partitioned_option.view == View.PROBLEM else View.AGENT
            if view == View.PROBLEM:
                states = partitioned_option.problem_states
            else:
                states = partitioned_option.agent_states
            self._links = [Link(states, None, **kwargs)]
        else:
            self._links = list()
            for subpartition in self._subpartitions:
                init_states = subpartition.states  # initial states based on subpartition's view
                for prob, _, _, next_states, _ in subpartition.effects():
                    self._links.append(Link(init_states, next_states, probability=prob, **kwargs))

        self.linking_function = defaultdict(list)

    @property
    def schemata(self) -> List[LinkedPDDLOperator]:
        return self._schemata

    @property
    def links(self):
        return self._links

    @property
    def option(self) -> int:
        return self._partitioned_option.option

    @property
    def partition(self) -> int:
        return self._partitioned_option.partition

    @property
    def n_subpartitions(self) -> int:

        if len(self._subpartitions) == 0:
            return 1
        return len(self._subpartitions)

    def add_problem_symbols(self, precondition_idx: int, effect_idx: int, prob: float):

        self.linking_function[precondition_idx].append((prob, effect_idx))

        for operator in self._schemata:
            operator.add_link(precondition_idx, effect_idx, prob)
