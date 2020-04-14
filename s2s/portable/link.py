import warnings

from sklearn.cluster import DBSCAN
from typing import List, Optional

from s2s.core.learned_operator import LearnedOperator
from s2s.core.partitioned_option import PartitionedOption
from s2s.pddl.pddl_operator import PDDLOperator
import numpy as np


class Link:

    def __init__(self, start: np.ndarray, end: Optional, probability=1, **kwargs):
        """
        Create a new link from the start state clusters to end clusters
        :param start: the start states
        :param end: the end states
        :param probability: the probability of transitioning from start to end states (accounting for probabilistic
        effects)
        """
        # end is already partitioned, but start may be stochastic. So check too
        self.end = end
        self.starts = self._partition(start, **kwargs)
        self._probability = probability

    def __iter__(self):
        for start in self.starts:
            # yield samples2np(start), None if self.end is None else samples2np(self.end)
            yield start, self.end, self._probability

    def _partition(self, X, **kwargs):
        # states = samples2np(X)

        min_samples = kwargs.get('init_min_samples', 1)
        epsilon = kwargs.get('effect_epsilon', 0.03)

        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
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
