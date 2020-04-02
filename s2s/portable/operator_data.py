from typing import List

from s2s.core.learned_operator import LearnedOperator
from s2s.core.partitioned_option import PartitionedOption
from s2s.pddl.pddl_operator import PDDLOperator

import numpy as np


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
        self._schemata = pddl_operators
        self._learned_operator = learned_operator

        if len(self._subpartitions) == 0:
            #  Then we need to ground the precondition only
            states = np.array([partitioned_option.extract_prob_space(partitioned_option.observations[i]) for i in
                               range(partitioned_option.states.shape[0])])
            self.links = [Link(states, None)]
        else:
            self.links = [Link(subpartition.states, subpartition.next_states) for subpartition in self._subpartitions]

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

    #  init set in problem space
    def observations(self, idx=0):
        return samples2np(self._subpartitions[idx].states)

    # terminal set in problem space
    def next_observations(self, idx=0):
        # problem space is just xy position (first 4)
        return samples2np(self._subpartitions[idx].next_states)

    def add_problem_symbols(self, pddl, precondition_idx, effect_idx):
        print("Adding p_symbol{} and p_symbol{}".format(precondition_idx, effect_idx))
        for operator in self._schemata:
            precondition = Predicate('psymbol_{}'.format(precondition_idx))
            if effect_idx != -1:
                effect = Predicate('psymbol_{}'.format(effect_idx))
                operator.link(precondition, effect)
            else:
                operator.link(precondition, None)
                # operator.add_effect(effect)
                # operator.add_effect(precondition.negate())

            # propositionalise objects to avoid ambiguity
            mask = self.full_mask
            instantiated = False
            for i, m in enumerate(mask):
                if not pddl.is_ambiguous(m):
                    # object is its own type, so can ignore!
                    continue
                else:
                    new_type = 'type{}{}'.format(pddl.object_type(m), chr(ord('a') + m - 1))
                    pddl.add_grounded_type(m, 'type{}'.format(pddl.object_type(m)), new_type)
                    operator.instantiate_object(i, new_type)
                    instantiated = True

            if not instantiated:
                operator.instantiate_object(-1, None)

    def observation_mask(self, idx):

        if len(self._subpartitions) == 0:
            return []

        masks = set()
        for obs, next_obs in zip(self.observations(idx), self.next_observations(idx)):
            mask = np.array([j for j in range(0, len(obs)) if not np.array_equal(obs[j], next_obs[j])])
            for m in mask:
                masks.add(m)
        return np.sort(list(masks))
