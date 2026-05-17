import numpy as np
import argparse


def _format_matrix_2d(matrix):
    """Render a matrix in a readable 2D block for logging."""
    return np.array2string(np.asarray(matrix), precision=6, suppress_small=True)


def _format_matrix_inline(matrix):
    """Render a matrix on one line for compact logging."""
    rows = [
        "[" + ", ".join(f"{value:.6g}" for value in row) + "]"
        for row in np.asarray(matrix)
    ]
    return "[" + "; ".join(rows) + "]"


def cholesky(matrix, verbose=False):
    """
    Compute the lower-triangular Cholesky factor of a symmetric positive definite matrix.

    Parameters
    - matrix: array-like, symmetric positive-definite matrix
    - verbose: bool, if True prints intermediate computation details
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    lower = np.zeros_like(matrix)
    computed = np.zeros_like(lower, dtype=bool)

    for i in range(n):
        if verbose:
            print(f"Processing row {i}")
        for j in range(i + 1):
            sum_prod = np.dot(lower[i, :j], lower[j, :j])
            value = matrix[i, j] - sum_prod
            if verbose:
                print(
                    f"  Computing L[{i},{j}]: A[{i},{j}]={matrix[i,j]}, sum_prod={sum_prod}, value={value}"
                )

            if i == j:
                if value <= 0:
                    raise np.linalg.LinAlgError("Matrix is not positive definite")
                lower[i, j] = np.sqrt(value)
                computed[i, j] = True
                if verbose:
                    print(f"  Diagonal L[{i},{j}] = sqrt({value}) = {lower[i,j]}")
            else:
                lower[i, j] = value / lower[j, j]
                computed[i, j] = True
                if verbose:
                    print(
                        f"  Off-diagonal L[{i},{j}] = {value} / L[{j},{j}] ({lower[j,j]}) = {lower[i,j]}"
                    )

            # Detailed logging: compute residual after each element assignment
            if verbose:
                display_lower = np.where(computed, lower, np.nan)
                display_upper = display_lower.T
                recon = lower @ lower.T
                residual = matrix - recon
                print("    L =")
                print(_format_matrix_2d(display_lower))
                print("    L^T =")
                print(_format_matrix_2d(display_upper))
                print("    L L^T + Residual = Matrix")
                print(_format_matrix_2d(recon))
                print("+")
                print(_format_matrix_2d(residual))
                print("=")
                print(_format_matrix_2d(matrix))
                print("")

    return lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cholesky decomposition demo")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show step-by-step process")
    args = parser.parse_args()

    # Example symmetric positive-definite matrix (classic example)
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)

    print("Input matrix A:")
    print(A)
    print()

    L = cholesky(A, verbose=args.verbose)

    print("Cholesky factor L (lower triangular):")
    print(L)
    print()
    print("Reconstructed A from L @ L.T:")
    print(L @ L.T)