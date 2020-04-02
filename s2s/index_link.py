from collections import defaultdict

from s2s.union_find import UnionFind


class IndexLink:
    """
    This is a way of associating two indices together. The association is two-way, and each index can be associated
    with many others.
    """

    def __init__(self):
        self._vals = defaultdict(set)

    def add(self, a: int, b: int):
        """
        Associate one index with another
        """
        self._vals[a].add(b)
        self._vals[b].add(a)

    def reduce(self, union_find: UnionFind):
        """
        Modify all the indices in place, given that they may have been unioned with others
        """
        temp = defaultdict(set)
        for k, vals in self._vals.items():
            temp[union_find[k]] = {union_find[x] for x in vals}
        self._vals = temp

    def __getitem__(self, index):
        return self._vals[index]
