import itertools

from typing import Iterable, Union

import numpy as np

from s2s.pddl.pddl import Proposition
from s2s.portable.problem_symbols import _FactoredProblemProposition, _ProblemProposition
from s2s.utils import range_without


class QuickCluster:

    def __init__(self, n_dims: int, threshold: float):
        self._n_dims = n_dims
        self._clusters = [list() for _ in range(n_dims)]
        self._threshold = threshold
        self._names = dict()

    def add(self, state: np.ndarray):

        for dim in range(self._n_dims):
            data = state[dim]
            found = False
            for i, (existing_mean, count) in enumerate(self._clusters[dim]):
                if abs(existing_mean - data) < self._threshold:
                    self._clusters[dim][i] = (existing_mean + (data - existing_mean) / (count + 1), count + 1)
                    found = True
                    break
            if not found:
                self._clusters[dim].append((data, 1))

    def get(self, state: np.ndarray, index_only=False) -> Union[Proposition, int]:
        indices = list()
        mean_state = list()
        for dim in range(self._n_dims):
            data = state[dim]
            idx = self._find_closest(data, self._clusters[dim])
            indices.append(idx)
            mean_state.append(self._clusters[dim][idx][0])

        if index_only:
            return self._get_name(indices)
        return _ProblemProposition(self._get_name(indices), np.array(mean_state))

    @property
    def factors(self):

        for dim in range(self._n_dims):
            for i, data in enumerate(self._clusters[dim]):
                temp = np.array(data[0])
                yield _FactoredProblemProposition(i, dim, np.reshape(temp, (1, 1)))

    def _exists(self, indices: Iterable[int]):
        id = int(''.join(map(str, indices)))
        return id in self._names

    @property
    def propositions(self):
        lengths = [range_without(0, len(x)) for x in self._clusters]
        for indices in itertools.product(*lengths):
            if not self._exists(indices):
                continue
            data = list()
            min_length = np.inf
            for dim in range(self._n_dims):
                temp = self._clusters[dim][indices[dim]]
                data.append(temp)
                min_length = min(min_length, len(temp))
            data = [x[0:min_length] for x in data]
            yield _ProblemProposition(self._get_name(indices), np.vstack(data).T)

    def __iter__(self):
        for x in self.factors:
            yield x
        for x in self.propositions:
            yield x

    def _get_name(self, indices: Iterable[int]) -> int:
        id = int(''.join(map(str, indices)))
        if id not in self._names:
            self._names[id] = len(self._names)
        return self._names[id]

    def __len__(self) -> int:
        return len(self._names)

    def _find_closest(self, data, existing):
        d = np.inf
        closest = -1
        for i, (e, _) in enumerate(existing):
            dist = abs(e - data)
            if dist < d:
                d = dist
                closest = i
        return closest
