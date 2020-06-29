from typing import Iterable, List

from s2s.env.s2s_env import S2SEnv
from s2s.pddl.pddl_operator import PDDLOperator
from s2s.pddl.pddl import Proposition
from s2s.pddl.unique_list import UniquePredicateList
from s2s.utils import indent


class PDDLDomain:

    def __init__(self, env: S2SEnv, vocabulary: UniquePredicateList, operators: List[PDDLOperator], **kwargs):
        self._env = env
        self._vocabulary = vocabulary
        self._operators = operators
        self._probabilistic = kwargs.get('probabilistic', True)
        self._rewards = kwargs.get('specify_rewards', True)
        self._conditional_effects = kwargs.get('conditional_effects', False)
        self.problem_symbols = set()

    @property
    def probabilistic(self):
        return self._probabilistic

    @probabilistic.setter
    def probabilistic(self, value):
        self._probabilistic = value

    @property
    def specify_rewards(self):
        return self._rewards

    @specify_rewards.setter
    def specify_rewards(self, value):
        self._rewards = value

    @property
    def conditional_effects(self):
        return self._conditional_effects

    @conditional_effects.setter
    def conditional_effects(self, value):
        self._conditional_effects = value

    def set_problem_symbols(self, problem_symbols):
        self.problem_symbols = problem_symbols

    def copy(self, keep_operators=True) -> 'PDDLDomain':
        """
        Makes a copy of the the PDDL domain
        :param keep_operators: whether the PDDL operators should be copied
        """
        operators = self._operators if keep_operators else []
        new_domain = PDDLDomain(self._env, self._vocabulary, operators, probabilistic=self.probabilistic,
                                specify_rewards=self.specify_rewards, conditional_effects=self.conditional_effects)
        new_domain.set_problem_symbols(self.problem_symbols)
        return new_domain

    @property
    def operators(self) -> List[PDDLOperator]:
        return self._operators

    def add_operator(self, operator: PDDLOperator):
        self._operators.append(operator)

    def add_operator(self, operator: PDDLOperator):
        self._operators.append(operator)

    def __str__(self):
        comment = ';Automatically generated {} domain PPDDL file.'.format(self._env.name)
        definition = 'define (domain {})'.format(self._env.name)
        requirements = '(:requirements :strips{}{}{})'.format(' :probabilistic-effects' if self.probabilistic else '',
                                                              ' :rewards' if self.specify_rewards else '',
                                                              ' :conditional-effects :fluents :equality :disjunctive-preconditions' if self.conditional_effects else '')

        symbols = '{}\n'.format(Proposition.not_failed()) + '\n'.join(
            ['{}'.format(x) for x in self._vocabulary])

        if self.conditional_effects:
            requirements += '\n\n(:functions (linking))'
        else:
            symbols += '\n\n' + '\n'.join(['(psymbol_{})'.format(name) for name in self.problem_symbols])

        predicates = '(:predicates\n{}\n)'.format(indent(symbols))

        format_spec = ':'
        if self._probabilistic:
            format_spec += 'p'
        if self._rewards:
            format_spec += 'r'

        operators = '\n\n'.join(
            [x.pretty_print(i, self._env.describe_option, probabilistic=self.probabilistic,
                            use_rewards=self.specify_rewards, conditional_effects=self.conditional_effects) for i, x in
             enumerate(self._operators)])

        description = '{}\n({}\n{}\n\n{}\n\n{}\n)'.format(
            comment,
            definition,
            indent(requirements),
            indent(predicates),
            indent(operators)
        )
        return description
