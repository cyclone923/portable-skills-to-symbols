from typing import Callable, Any, Tuple

from s2s.pddl.pddl import Proposition
from s2s.utils import if_not_none, range_without
import numpy as np


class _ProblemProposition(Proposition):
    """
    Holder class used for visualisation
    """

    def __init__(self, index, data: np.ndarray):
        super().__init__('psymbol_{}'.format(index), None)
        self.data = data

    @property
    def mask(self):
        return range_without(0, self.data.shape[1])

    def sample(self, n_samples):
        return self.data[np.random.randint(self.data.shape[0], size=n_samples), :]



class ProblemSymbols:
    """
    A class that holds a set of problem-specific symbols.
    """

    def __init__(self, similarity_check: Callable[[Any, Any], bool] = None):
        self._symbols = list()
        self._similarity_check = if_not_none(similarity_check, self._is_similar)

    def __len__(self) -> int:
        return len(self._symbols)

    def add(self, symbol: Any) -> int:
        """
        Add a symbol to the list, accounting for duplicates
        :param symbol: the symbol to add
        :return: the index of the symbol in the list. If symbol to be added is similar to an existing one, the
        symbol is not added to the list, and the index of the existing one is returned
        """
        for i, existing_symbols in enumerate(self._symbols):
            if self._similarity_check(symbol, existing_symbols):
                return i
        self._symbols.append(symbol)
        return len(self._symbols) - 1

    def _is_similar(self, x, y):
        mean = np.mean(x, axis=0)
        m = np.mean(y, axis=0)
        return np.linalg.norm(mean - m, np.inf) < 0.1

    def means(self):
        return np.array([np.mean(data, axis=0) for data in self._symbols])

    def __iter__(self):
        for i, data in enumerate(self._symbols):
            yield _ProblemProposition(i, data)
