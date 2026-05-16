import numpy as np


def cholesky(matrix):
    """
    Compute the lower-triangular Cholesky factor of a symmetric positive definite matrix.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    lower = np.zeros_like(matrix)

    for i in range(n):
        for j in range(i + 1):
            value = matrix[i, j] - np.dot(lower[i, :j], lower[j, :j])

            if i == j:
                if value <= 0:
                    raise np.linalg.LinAlgError("Matrix is not positive definite")
                lower[i, j] = np.sqrt(value)
            else:
                lower[i, j] = value / lower[j, j]

    return lower