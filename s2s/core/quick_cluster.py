from typing import Iterable

import numpy as np


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

    def get(self, state: np.ndarray):
        indices = list()
        for dim in range(self._n_dims):
            data = state[dim]
            indices.append(self._find_closest(data, self._clusters[dim]))
        return self._get_name(indices)

    def _get_name(self, indices: Iterable[int]):
        id = int(''.join(map(str, indices)))
        if id not in self._names:
            self._names[id] = len(self._names)
        return self._names[id]

    def __len__(self) -> int:
        return len(self._names)

    def _find_closest(self, data, existing):
        d = np.inf
        closest = -1
        for i, e in enumerate(existing):
            dist = abs(e - data)
            if dist < d:
                d = dist
                closest = i
        return closest
