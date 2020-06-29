import copy
from typing import Optional, List, Union, Tuple

from s2s.estimators.estimators import StateDensityEstimator
from s2s.utils import if_not_none, indent


class Proposition:
    """
    A non-typed, non-lifted predicate (i.e. a proposition)
    """

    def __init__(self, name: str, kde: Optional[StateDensityEstimator]):
        self._name = name
        self._kde = kde
        self.sign = 1  # whether true or the negation of the predicate
        self._noop = kde is not None and kde.is_noop

    @property
    def is_noop(self):
        return self._noop

    @property
    def estimator(self) -> StateDensityEstimator:
        return self._kde

    @property
    def name(self) -> str:
        return self._name

    @property
    def mask(self):
        return self._kde.mask

    def sample(self, n_samples):
        return self.estimator.sample(n_samples)

    @property
    def is_grounded(self) -> bool:
        return False

    def negate(self) -> 'Proposition':
        """"
        Creates a copy of the predicate, negated
        """
        clone = copy.copy(self)
        clone.sign *= -1
        return clone

    def __str__(self):
        if self.sign < 0:
            return '(not ({}))'.format(self.name)
        return '({})'.format(self.name)

    @staticmethod
    def not_failed():
        return Proposition("notfailed", None)


class FluentPredicate(Proposition):

    def __init__(self, operator: str, name: str, value: float):
        super().__init__(name, None)
        self._operator = operator
        self._value = value

    def __str__(self):
        return '({} ({}) {})'.format(self._operator, self.name, self._value)

    @property
    def is_noop(self):
        return False


class RewardPredicate(FluentPredicate):
    """
    A proposition that handles changing reward
    """

    def __init__(self, predicted_reward: float):
        """
        Create a new reward proposition
        :param predicted_reward: teh predicted increase or decrease in reward
        """
        operator = 'increase' if predicted_reward >= 0 else 'decrease'
        super().__init__(operator, 'reward', abs(round(predicted_reward, 2)))


class Clause:
    """
    A collection of propositions
    """

    def __init__(self, symbols: List[Proposition] = None, conjunctive=True):
        """
        Create a new clause with associated probability
        :param symbols: the propositions
        :type conjunctive: True if the propositions are combined conjunctively, false if disjunctive
        """
        self._conjunctive = conjunctive
        symbols = if_not_none(symbols, list())
        self._symbols = copy.copy(symbols)

    def add(self, other: Proposition) -> 'Clause':
        return self.__add__(other)

    def __add__(self, other: Proposition) -> 'Clause':
        self._symbols.append(other)
        return self

    def has_effect(self):
        for x in self._symbols:
            if not x.is_noop:
                return True
        return False

    def __str__(self) -> str:
        propositions = [x for x in self._symbols if not x.is_noop]
        if len(propositions) == 0:
            raise ValueError("No propositions found")
        if len(propositions) == 1:
            return '{}'.format(propositions[0])
        conjunction = 'and' if self._conjunctive else 'or'
        return '({} {})'.format(conjunction, ' '.join(['{}'.format(x) for x in propositions]))


class Probabilistic:
    """
    A holder for probabilistic clauses
    """

    def __init__(self):
        self.prob_sum = 0
        self._values: List[Tuple[float, Clause]] = list()

    def add(self, value: Union[Clause, Proposition], prob=1) -> 'Probabilistic':
        if isinstance(value, Proposition):
            value = Clause([value])  # turn it into a clause

        self._values.append((prob, value))
        self.prob_sum += prob
        return self

    def __len__(self):
        return len(self._values)

    def __str__(self):

        if len(self._values) == 1:
            return str(self._values[0][1])

        effect = '(probabilistic'
        for prob, value in self._values:
            if value.has_effect():
                prob = round(prob / self.prob_sum, 3)  # TODO probably a better way!
                effect += indent('\n{} {}'.format(prob, value), 2)
        effect += ')'
        return effect


class ConditionalEffect:

    def __init__(self, precondition: Union[Proposition, Clause], effect: Union[Clause, Probabilistic]):
        self._precondition = precondition
        self._effect = effect

    def __str__(self):
        return '(when {} {})'.format(self._precondition, self._effect)


class MixedEffect:
    """
    Holder for effects that are both regular and conditional
    """

    def __init__(self, regular_effect: Union[Clause, Probabilistic], conditional_effects: List[ConditionalEffect]):
        self._regular_effect = regular_effect
        self._conditional_effects = conditional_effects

    def __len__(self):
        return len(self._conditional_effects) + 1 if isinstance(self._regular_effect, Clause) else len(
            self._regular_effect)

    def __str__(self):
        return '(and {}\n{})'.format(self._regular_effect,
                                         '\n'.join([indent('{}'.format(x), 3) for x in self._conditional_effects]))
