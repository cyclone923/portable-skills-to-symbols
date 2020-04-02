from typing import List

from s2s.core.learned_operator import LearnedOperator
from s2s.pddl.pddl_operator import PDDLOperator
from s2s.pddl.proposition import Proposition
from s2s.utils import indent
from s2s.pddl.pddl_operator import _PrettyPrint as PP


class LinkedOperator(PDDLOperator):

    def __init__(self, learned_operator: LearnedOperator, name: str = None, task: int = None):
        """
        Create a new PDDL operator
        :param learned_operator: the estimated operator
        :param name: the name of the operator (optional)
        :param task: the task ID (ignore this if there is only one task)
        """
        super().__init__(learned_operator, name, task)
        self._links = dict()

    def add_link(self, start, end):
        # TODO assuming determinstic for now
        self._links[start] = end

    def pretty_print(self, index=None, probabilistic=True, use_rewards=True, conditional_effects=True,
                     option_descriptor=None):
        """
        Print everything out nicely
        """
        return str(_PrettyPrint(self, index, probabilistic, use_rewards, conditional_effects, option_descriptor))


class _PrettyPrint(PP):

    def __init__(self, operator: PDDLOperator, index=None, probabilistic=True, conditional_effects=True, use_rewards=True,
                 option_descriptor=None):
        super().__init__(operator, index, probabilistic, use_rewards, option_descriptor)
        self._conditional_effects = conditional_effects

    def __str__(self):
        precondition = self._propositions_to_str(self._operator.preconditions)

        if self._probabilistic:
            effects = self._operator.effects
        else:
            effects = [max(self._operator.effects, key=lambda x: x[0])]  # get most probable

        if len(effects) == 1:
            end = None
            if self._use_rewards and effects[0][2] is not None:
                end = '{} (reward) {:.2f}'.format('increase' if effects[0][2] >= 0 else 'decrease',
                                                  abs(effects[0][2]))
            effect = self._propositions_to_str(effects[0][1], end=end)
        else:
            effect = 'probabilistic '

            total_prob = sum(prob for prob, _, _ in effects)  # sometimes total prob is just over 1 because rounding :(

            for prob, eff, reward in effects:

                prob = round(prob / total_prob, 3)  # TODO probably a better way!
                end = None
                if self._use_rewards and reward is not None:
                    end = '{} (reward) {:.2f}'.format('increase' if reward >= 0 else 'decrease', abs(reward))
                effect += indent('\n\t{} ({})'.format(prob, self._propositions_to_str(eff, end)), 3)
            effect += '\n\t\t\t\n\t'

        if self._option_descriptor is None:
            name = self._operator.name
        else:
            name = '{}-partition-{}'.format(self._option_descriptor(self._operator.option), self._operator.partition)
        if self._index is not None:
            name += '-{}'.format(self._index)

        return '(:action {}\n\t:parameters ()\n\t:precondition ({})\n\t:effect ({})\n)'.format(name,
                                                                                               precondition,
                                                                                               effect)

    def _propositions_to_str(self, propositions: List[Proposition], end=None) -> str:
        if len(propositions) == 0:
            raise ValueError("No propositions found")

        propositions = list(map(str, propositions))
        if end is not None:
            propositions.append(end)

        if len(propositions) == 1:
            return '{}'.format(propositions[0])
        return 'and {}'.format(' '.join(['({})'.format(x) for x in propositions]))
