from collections import defaultdict

from typing import List
import numpy as np
from s2s.estimators.estimators import PreconditionClassifier
from s2s.utils import is_single_sample


class SimpleClassifier(PreconditionClassifier):
    """
    An implementation of a simple classifier that uses a count-based approach and checks for equality
    """

    def __init__(self, mask: List[int], probabilistic=True):
        """
        Create a new SVM classifier for preconditions
        :param mask: the state variables that should be kept for classification
        :param probabilistic: whether the classifier is probabilistic
        """
        self._mask = mask
        self._probabilistic = probabilistic
        self._classifier = defaultdict(float)

    @property
    def mask(self) -> List[int]:
        """
        Get the precondition mask
        """
        return self._mask

    def _convert(self, x: np.ndarray):
        # TODO make general
        return tuple(map(int, x))

    def fit(self, X, y, verbose=False, **kwargs):
        """
        Fit the data to the classifier using a grid search for the hyperparameters with cross-validation
        :param X: the data
        :param y: the labels
        :param verbose: the verbosity level
        """
        positives = defaultdict(int)
        negatives = defaultdict(int)
        for x, label in zip(X, y):
            key = self._convert(x)
            if label:
                positives[key] += 1
            else:
                negatives[key] += 1

        for x, n_positive in positives.items():
            n_negative = negatives[x]
            self._classifier[x] = n_positive / (n_positive + n_negative)

    def probability(self, states: np.ndarray) -> float:
        """
        Compute the probability of the state given the learned classifier
        :param states: the states
        :return: the probability of the state according to the classifier
        """
        if is_single_sample(states):
            states = states.reshape(1, -1)
        masked_states = states[:, self.mask]

        prob = 0
        for x in masked_states:
            prob += self._classifier[self._convert(x)]
        return prob / len(masked_states)