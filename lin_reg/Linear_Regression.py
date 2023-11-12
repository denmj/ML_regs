import numpy as np
import scipy.linalg as la

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
    print(f'A shape: {A.shape}')
    b = X.T.dot(Y)
    print(f'b shape: {b.shape}')
    theta = np.zeros_like(X.shape[1])
    print(f'theta shape: {theta.shape}')

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

def lu_decomposition_simple(X: np.ndarray, Y: np.ndarray) -> np.ndarray:

    """Use scipy """
    A = X.T.dot(X)
    b = X.T.dot(Y)
    theta = np.zeros_like(X.shape[1])
    P, L, U = la.lu(A)
    y = la.solve_triangular(L, b, lower=True)
    theta = la.solve_triangular(U, y)
    return theta

# Method 3: QR Decomposition 
def qr_decomposition(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Return the solution to the linear regression problem using QR decomposition
    """
    Q, R = np.linalg.qr(X)
    return np.linalg.inv(R).dot(Q.T).dot(Y)


# Method 4: SVD Decomposition
def svd_decomposition(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Return the solution to the linear regression problem using SVD decomposition
    """
    U, S, V = np.linalg.svd(X)
    return V.T.dot(np.linalg.inv(np.diag(S))).dot(U.T).dot(Y)


# Method 5: Cholesky Decomposition
def cholesky_decomposition(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Return the solution to the linear regression problem using Cholesky decomposition
    """
    L = np.linalg.cholesky(X.T.dot(X))
    return np.linalg.inv(L.T).dot(np.linalg.inv(L)).dot(X.T).dot(Y)

