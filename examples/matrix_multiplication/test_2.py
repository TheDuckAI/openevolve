import itertools
import math
import random
from typing import List, Tuple

from tqdm import tqdm


# ---------------------------------------------------------------------
# 1.  Build the reference 4×4×4×4×4×4 tensor in flattened (16,16,16) form
# ---------------------------------------------------------------------
def matmul_tensor_4() -> List[int]:
    T = [[[0 for _ in range(16)] for _ in range(16)] for _ in range(16)]
    for i, j, k in itertools.product(range(4), repeat=3):
        a = 4 * i + j  #  A_{ij}  flattened
        b = 4 * j + k  #  B_{jk}
        c = 4 * i + k  #  C_{ik}
        T[a][b][c] = 1
    return T  # nested lists are fastest to mutate in pure Python


TARGET = matmul_tensor_4()


# ---------------------------------------------------------------------
# 2.  Helpers
# ---------------------------------------------------------------------
def rank1(u: List[int], v: List[int], w: List[int]) -> List[List[List[int]]]:
    """Outer product; returns 16³ list-cube."""
    return [[[u[a] * v[b] * w[c] for c in range(16)] for b in range(16)] for a in range(16)]


def add_inplace(acc, term, sign=1):
    for a in range(16):
        for b in range(16):
            row_acc, row_term = acc[a][b], term[a][b]
            for c in range(16):
                row_acc[c] += sign * row_term[c]


def l2_error(acc) -> int:
    """Squared L₂ error against TARGET (all integers, so int is fine)."""
    err = 0
    for a in range(16):
        for b in range(16):
            row_acc, row_tar = acc[a][b], TARGET[a][b]
            for c in range(16):
                diff = row_acc[c] - row_tar[c]
                err += diff * diff
    return err


# ---------------------------------------------------------------------
# 3.  Search parameters
# ---------------------------------------------------------------------
RANK = 64  # start with Strassen-style rank; drop later
VALUES = [-1, 0, 1]  # allowed entries (small keeps search tractable)
STEPS = 120_000  # ~2–3 min on a 3 GHz laptop
TEMPERATURE_0 = 2.0  # simulated-annealing schedule


# ---------------------------------------------------------------------
# 4.  Search
# ---------------------------------------------------------------------
def find_decomposition(seed: int = 0) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    random.seed(seed)

    # Initialise factors with i.i.d. {-1,0,1}
    U = [[random.choice(VALUES) for _ in range(16)] for _ in range(RANK)]
    V = [[random.choice(VALUES) for _ in range(16)] for _ in range(RANK)]
    W = [[random.choice(VALUES) for _ in range(16)] for _ in range(RANK)]

    tensor_acc = [[[0] * 16 for _ in range(16)] for _ in range(16)]
    for r in range(RANK):
        add_inplace(tensor_acc, rank1(U[r], V[r], W[r]))

    best_err = l2_error(tensor_acc)

    for step in tqdm(range(1, STEPS + 1)):
        # Annealing temperature
        T = TEMPERATURE_0 * (1 - step / STEPS)

        # Pick random (rank-slot, factor, coordinate) and propose ±1 perturbation
        r = random.randrange(RANK)
        xyz = random.choice(("u", "v", "w"))
        i = random.randrange(16)

        factor = {"u": U, "v": V, "w": W}[xyz]
        old_val = factor[r][i]
        new_val = old_val + random.choice([-1, 1])
        if new_val not in VALUES:  # stay in the grid
            continue

        # Δ-update the accumulated tensor
        add_inplace(tensor_acc, rank1(U[r], V[r], W[r]), sign=-1)  # remove old
        factor[r][i] = new_val
        add_inplace(tensor_acc, rank1(U[r], V[r], W[r]), sign=+1)  # add new

        # Metropolis accept / reject
        err = l2_error(tensor_acc)
        if err < best_err or random.random() < math.exp(-(err - best_err) / max(T, 1e-6)):
            best_err = err
            if best_err == 0:
                print(f"Exact decomposition found at step {step}")
                break
        else:
            # undo
            add_inplace(tensor_acc, rank1(U[r], V[r], W[r]), sign=-1)
            factor[r][i] = old_val
            add_inplace(tensor_acc, rank1(U[r], V[r], W[r]), sign=+1)

    return U, V, W


# ---------------------------------------------------------------------
# 5.  Verification (exact equality after rounding, as AlphaEvolve does)
# ---------------------------------------------------------------------
def verify(U, V, W):
    acc = [[[0] * 16 for _ in range(16)] for _ in range(16)]
    for r in range(RANK):
        add_inplace(acc, rank1(U[r], V[r], W[r]))
    ok = acc == TARGET
    print("Verification:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------
# 6.  Entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    decomp = find_decomposition()
    verify(*decomp)
