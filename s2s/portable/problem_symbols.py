from typing import Callable, Any, Tuple

from s2s.utils import if_not_none
import numpy as np


class ProblemSymbols:
    """
    A class that holds a set of problem-specific symbols.
    """

    def __init__(self, similarity_check: Callable[[Any, Any], bool] = None):
        self._symbols = list()
        self._similarity_check = if_not_none(similarity_check, self.__is_similar)

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

    def __is_similar(self, x, y):
        mean = np.mean(x, axis=0)
        m = np.mean(y, axis=0)
        return np.linalg.norm(mean - m, np.inf) < 0.55

    def means(self):
        return np.array([np.mean(data, axis=0) for data in self._symbols])
