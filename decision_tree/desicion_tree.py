# Desicion tree implementation

import numpy as np

class DecisionTree(object):

    def __init__(self):
        pass

    def entropy(self, y):
        """
        Calculate the entropy of a dataset.

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        entropy : float
            Entropy of y.
        """

        probs = np.bincount(y) / len(y)

       # calculate entropy
        entropy = -probs.dot(np.log2(probs))

        return entropy
    
    def best_split(self, X, y):
        """
        Find the best split for a dataset.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]

        Returns
        -------
        best_feature : int
            Index of the best feature.
        best_info_gain : float
            Information gain of the best split.

        """

        base_entropy = self.entropy(y)
        best_info_gain = 0
        best_feature = None

        n_features = X.shape[1]

        for feature in range(n_features):
            pass


        return best_feature, best_info_gain