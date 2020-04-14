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

    def __init__(self, symbols: List[Proposition] = None):
        """
        Create a new clause with associated probability
        :param symbols: the propositions
        """
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
        return '(and {})'.format(' '.join(['{}'.format(x) for x in propositions]))


class Probabilistic:
    """
    A holder for probabilistic clauses
    """

    def __init__(self):
        self.prob_sum = 0
        self._values: List[Tuple[float, Clause]] = list()

    def add(self, value: Clause, prob=1) -> 'Probabilistic':
        self._values.append((prob, value))
        self.prob_sum += prob
        return self

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
        precondition = indent(str(self._precondition), 2)
        effect = indent(str(self._effect), 2)
        return '(when\n{}\n{}\n)'.format(precondition, effect)
