import numpy as np

from typing import List


# Method 1: Matrix Inversion (Normal Equation)
def normal_equation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Returns the solution to the linear regression problem using the normal equation method.

    Parameters
    ----------
    X : np.ndarray
        The design matrix.

    Y : np.ndarray
        The response vector.
    
    Returns
    -------
    np.ndarray
        The solution to the linear regression problem.

    """

    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))

