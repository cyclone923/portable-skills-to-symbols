import itertools

from typing import Callable, Any, Tuple, List, Iterable

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


class _FactoredProblemProposition(Proposition):
    """
    Holder class used for visualisation
    """

    def __init__(self, idx, dim, data: np.ndarray):
        super().__init__('factor_dim_{}_{}'.format(dim, idx), None)
        self.data = data
        self.dim = dim

    @property
    def mask(self):
        return [self.dim]

    def sample(self, n_samples):
        return np.reshape(self.data[np.random.randint(self.data.shape[0], size=n_samples)], (n_samples, 1))


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


class FactoredProblemSymbols:
    """
    A class that holds a set of problem-specific symbols.
    """

    def __init__(self, n_dims: int, similarity_check: Callable[[int, np.ndarray, np.ndarray], bool] = None):
        self._symbols = [list() for _ in range(n_dims)]
        self.n_dims = n_dims
        self._similarity_check = if_not_none(similarity_check, self._is_similar)
        self._names = dict()

    def __len__(self) -> int:
        return len(self._names)

    def add(self, symbol: np.ndarray) -> int:
        """
        Add a symbol to the list, accounting for duplicates
        :param symbol: the symbol to add
        :return: the index of the symbol in the list. If symbol to be added is similar to an existing one, the
        symbol is not added to the list, and the index of the existing one is returned
        """
        indices = list()
        for dim in range(symbol.shape[1]):
            data = symbol[:, dim]
            found = False
            for i, existing_symbols in enumerate(self._symbols[dim]):
                if self._similarity_check(dim, data, existing_symbols):
                    indices.append(i)
                    found = True
                    break
            if not found:
                indices.append(len((self._symbols[dim])))
                self._symbols[dim].append(data)
        return self._get_name(indices)

    def _get_name(self, indices: Iterable[int]):
        id = int(''.join(map(str, indices)))
        if id not in self._names:
            self._names[id] = len(self._names)
        return self._names[id]

    def _exists(self, indices: Iterable[int]):
        id = int(''.join(map(str, indices)))
        return id in self._names

    def _is_similar(self, dim, x, y):
        mean = np.mean(x, axis=0)
        m = np.mean(y, axis=0)

        threshold = 0.25
        if dim == 2:
            threshold = 0.3
        return abs(mean - m) < threshold
        # return np.linalg.norm(mean - m, np.inf) < 0.1

    def means(self):
        return np.array([np.mean(data, axis=0) for data in self._symbols])

    def __iter__(self):

        for dim in range(len(self._symbols)):
            for i, data in enumerate(self._symbols[dim]):
                yield _FactoredProblemProposition(i, dim, data)

        lengths = [range_without(0, len(x)) for x in self._symbols]
        for indices in itertools.product(*lengths):

            if not self._exists(indices):
                continue

            data = list()
            min_length = np.inf
            for dim in range(len(self._symbols)):
                temp = self._symbols[dim][indices[dim]]
                data.append(temp)
                min_length = min(min_length, len(temp))

            data = [x[0:min_length] for x in data]

            yield _ProblemProposition(self._get_name(indices), np.vstack(data).T)

        #
