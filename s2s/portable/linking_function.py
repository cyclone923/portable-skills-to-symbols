from collections import defaultdict, Counter


class LinkingFunction:

    def __init__(self):
        self._sizes = defaultdict(int)
        self._mapping = defaultdict(Counter)

    def clear(self):
        self._sizes.clear()
        self._mapping.clear()

    def add(self, start: int, end: int):
        self._sizes[start] += 1
        self._mapping[start][end] += 1

    def __iter__(self):
        for start, counts in self._mapping.items():
            for end, count in counts.items():
                yield start, end, count / self._sizes[start]




