from typing import List

from s2s.core.learned_operator import LearnedOperator
from s2s.core.partitioned_option import PartitionedOption
from s2s.pddl.pddl_operator import PDDLOperator
import numpy as np


class Link:
    def __init__(self, start, end):

        # end is already partitioned, but start may be stochastic. So check too
        self.end = end
        self.starts = self._partition(start)

    def __iter__(self):
        for start in self.starts:
            yield samples2np(start), None if self.end is None else samples2np(self.end)

    def _partition(self, X):
        states = samples2np(X)
        epsilon = 0.5
        min_samples = max(3, min(10, len(states) // 10))  # at least 3, at most 10

        min_samples = 1

        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(states)
        labels = db.labels_
        if all(elem == -1 for elem in labels) and len(labels) > min_samples:
            warnings.warn("All datapoints classified as noise!")
            labels = [0] * len(labels)
        clusters = list()
        for label in set(labels):
            if label == -1:
                continue
            # Get the data belonging to the current cluster
            indices = [i for i in range(0, len(labels)) if labels[i] == label]
            clusters.append(X[indices])
        return clusters

