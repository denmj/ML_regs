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

    def information_gain(self, y, y_left, y_right):
        """
        Calculate the information gain.

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.
        y_left : array-like, shape = [n_samples]
            The target values of the left child node.
        y_right : array-like, shape = [n_samples]
            The target values of the right child node.

        Returns
        -------
        info_gain : float
            Information gain.
        """

        # calculate the parent entropy
        parent_entropy = self.entropy(y)

        # calculate the child entropy
        child_entropy = 0
        for child in [y_left, y_right]:
            child_entropy += self.entropy(child) * len(child) / len(y)

        # calculate the information gain
        info_gain = parent_entropy - child_entropy

        return info_gain
    
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
                
                # get all the unique values of the feature
                unique_values = np.unique(X[:, feature])
    
                # calculate the information gain
                new_entropy = 0
                for value in unique_values:
                    y_left = y[X[:, feature] == value]
                    y_right = y[X[:, feature] != value]
                    info_gain = self.information_gain(y, y_left, y_right)
                    new_entropy += info_gain
    
                # update the information gain
                if new_entropy > best_info_gain:
                    best_info_gain = new_entropy
                    best_feature = feature


        return best_feature, best_info_gain