import copy

from s2s.estimators.estimators import StateDensityEstimator


class Proposition:
    """
    A non-typed, non-lifted predicate (i.e. a proposition)
    """

    def __init__(self, name: str, kde: StateDensityEstimator):
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
        return self._kde.sample(n_samples)

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
            return 'not ({})'.format(self.name)
        return self.name

    @staticmethod
    def not_failed():
        return Proposition("notfailed", None)
