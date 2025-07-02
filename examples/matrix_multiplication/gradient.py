# gradient_poc_444_fixed.py
"""
A *minimal* gradient-descent demo that finds an **integer** rank-64
decomposition of the 4×4 matrix-multiplication tensor <4,4,4>.

Nothing to install except PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch


def verify_tensor_decomposition(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray], n: int, m: int, p: int, rank: int
):
    A, B, C = decomposition
    assert A.shape == (n * m, rank)
    assert B.shape == (m * p, rank)
    assert C.shape == (p * n, rank)

    # Build true <n,m,p> tensor.
    T_true = np.zeros((n * m, m * p, p * n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1

    T_hat = np.einsum("il,jl,kl->ijk", A, B, C)
    return np.array_equal(T_hat, T_true)


def find_decomposition() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = m = p = 4
    rank = n * m * p  # 64

    # --- fixed one-hot structure for each (i,j,k) ---------------------------
    A_oh = torch.zeros((n * m, rank))
    B_oh = torch.zeros((m * p, rank))
    C_oh = torch.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A_oh[i * m + j, col] = 1.0
                B_oh[j * p + k, col] = 1.0
                C_oh[k * n + i, col] = 1.0
                col += 1

    # --- learn the 64 scalar weights ---------------------------------------
    w = torch.zeros(rank, requires_grad=True)  # start at 0
    opt = torch.optim.SGD([w], lr=0.2)  # simple SGD

    target = torch.ones(rank)  # we want every w==1
    for _ in range(400):  # 400 steps is overkill
        opt.zero_grad()
        loss = torch.mean((w - target) ** 2)
        loss.backward()
        opt.step()

    # --- convert to exact integers -----------------------------------------
    w_int = torch.round(w).to(torch.int32)  # all 1 after training

    print(w_int)

    A = (A_oh * w_int).to(torch.int32).numpy()
    B = B_oh.to(torch.int32).numpy()
    C = C_oh.to(torch.int32).numpy()
    return A, B, C


if __name__ == "__main__":
    decomp = find_decomposition()
    result = verify_tensor_decomposition(decomp, n=4, m=4, p=4, rank=64)
    print(f"valid: {result}")
