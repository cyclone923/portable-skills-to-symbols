from collections import defaultdict, Counter
from typing import Generator, Tuple


class LinkingFunction:
    """
    A linking function is simply a function that maps one problem-specific proposition to another
    """

    def __init__(self):
        """
        Create an empty linking function
        """
        self._sizes = defaultdict(int)  # this will hold the number of times a start partition is added
        self._mapping = defaultdict(Counter)  # this will map start x end -> count

    def clear(self) -> None:
        """
        Clear the linking function
        """
        self._sizes.clear()
        self._mapping.clear()

    def add(self, start: int, end: int) -> None:
        """
        Add a start and end partition transition
        :param start: the start partition
        :param end: the end partition
        """
        self._sizes[start] += 1
        self._mapping[start][end] += 1

    def __iter__(self) -> Generator[Tuple[int, int, float], None, None]:
        """
        Yields each start-end link with its associated probability of occurring
        """
        for start, counts in self._mapping.items():
            for end, count in counts.items():
                yield start, end, count / self._sizes[start]

    def __contains__(self, start_partition: int) -> bool:
        """
        Checks whether a start index is present in the linking function
        :param start_partition: the start partition
        """
        return start_partition in self._sizes

    def __getitem__(self, start_partition: int) -> Generator[Tuple[int, float], None, None]:
        """
        Yield each start-end link with its associated probability of occurring
        :param start_partition: the start partition
        """
        for end, count in self._mapping[start_partition].items():
            yield end, count / self._sizes[start_partition]



