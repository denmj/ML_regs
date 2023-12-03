import numpy as np

import numpy as np
from decision_tree import DecisionTree

def bootstrap_sample_idx(x, y, n_samples):
    """Bootstrap sample of data"""
    samples = []
    for i in range(n_samples):
        idx = np.random.randint(0, len(x), len(x))
        samples.append((x[idx], y[idx]))

    return samples