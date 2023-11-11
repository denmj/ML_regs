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

# Method 2: LU Decomposition

def lu_decomposition(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Return the solution to the linear regression problem using A = LU 
    
    A needs to be a square matrix 
    """
    A = X.T.dot(X)
    b = X.T.dot(Y)
    theta = np.zeros_like(X.shape[1])

    n = A.shape[0]
    y = np.zeros(n)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    
    # Getting L and U
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_upper = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_upper

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal as 1
            else:
                sum_lower = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_lower) / U[i][i]

    # Forward subst
    for i in range(n):
        sum_j = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_j) / L[i][i]

    # Back subst
    for i in range(n-1, -1, -1):
        sum_j = sum(U[i][j] * theta[j] for j in range(i+1, n))
        theta[i] = (y[i] - sum_j) / U[i][i]
    print(1)
    return theta

# Method 3: QR Decomposition 
def qr_decomposition(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Return the solution to the linear regression problem using QR decomposition
    """
    Q, R = np.linalg.qr(X)
    return np.linalg.inv(R).dot(Q.T).dot(Y)

