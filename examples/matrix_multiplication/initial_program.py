# simple_444_decomposition.py
import numpy as np


def find_decomposition():
    """
    Return a rank-64 integer CP-decomposition of the <4,4,4> matrix-
    multiplication tensor.  Each column encodes one elementary product
    A[i,j] · B[j,k] contributing to C[i,k].
    """
    n = m = p = 4
    rank = n * m * p  # 64
    A = np.zeros((n * m, rank), dtype=np.int32)
    B = np.zeros((m * p, rank), dtype=np.int32)
    C = np.zeros((p * n, rank), dtype=np.int32)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[i * m + j, col] = 1  # position of A[i,j]
                B[j * p + k, col] = 1  # position of B[j,k]
                C[k * n + i, col] = 1  # position of C[i,k]
                col += 1

    return A, B, C


# @title Verification function


def verify_tensor_decomposition(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray], n: int, m: int, p: int, rank: int
):
    """Verifies the correctness of the tensor decomposition.

    Args:
      decomposition: a tuple of 3 factor matrices with the same number of columns.
        (The number of columns specifies the rank of the decomposition.) To
        construct a tensor, we take the outer product of the i-th column of the
        three factor matrices, for 1 <= i <= rank, and add up all these outer
        products.
      n: the first parameter of the matrix multiplication tensor.
      m: the second parameter of the matrix multiplication tensor.
      p: the third parameter of the matrix multiplication tensor.
      rank: the expected rank of the decomposition.

    Raises:
      AssertionError: If the decomposition does not have the correct rank, or if
      the decomposition does not construct the 3D tensor which corresponds to
      multiplying an n x m matrix by an m x p matrix.
    """
    # Check that each factor matrix has the correct shape.
    factor_matrix_1, factor_matrix_2, factor_matrix_3 = decomposition
    assert factor_matrix_1.shape == (n * m, rank), (
        f"Expected shape of factor matrix 1 is {(n * m, rank)}. Actual shape is {factor_matrix_1.shape}."
    )
    assert factor_matrix_2.shape == (m * p, rank), (
        f"Expected shape of factor matrix 1 is {(m * p, rank)}. Actual shape is {factor_matrix_2.shape}."
    )
    assert factor_matrix_3.shape == (p * n, rank), (
        f"Expected shape of factor matrix 1 is {(p * n, rank)}. Actual shape is {factor_matrix_3.shape}."
    )

    # Form the matrix multiplication tensor <n, m, p>.
    matmul_tensor = np.zeros((n * m, m * p, p * n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                matmul_tensor[i * m + j][j * p + k][k * n + i] = 1

    # Check that the tensor is correctly constructed.
    constructed_tensor = np.einsum("il,jl,kl -> ijk", *decomposition)
    assert np.array_equal(constructed_tensor, matmul_tensor), (
        f"Tensor constructed by decomposition does not match the matrix multiplication tensor <{(n, m, p)}>: {constructed_tensor}."
    )
    print(
        f"Verified a decomposition of rank {rank} for matrix multiplication tensor <{n},{m},{p}>."
    )

    # Print the set of values used in the decomposition.
    np.set_printoptions(linewidth=100)
    print(
        "This decomposition uses these factor entries:\n",
        np.array2string(
            np.unique(np.vstack((factor_matrix_1, factor_matrix_2, factor_matrix_3))),
            separator=", ",
        ),
    )


if __name__ == "__main__":
    decomp = find_decomposition()
    verify_tensor_decomposition(decomp, n=4, m=4, p=4, rank=64)
