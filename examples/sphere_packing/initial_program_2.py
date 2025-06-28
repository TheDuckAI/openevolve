# EVOLVE-BLOCK-START
"""Advanced hybrid optimisation for packing 26 unequal spheres in a 1×1×1 cube.

The routine:
1.  Builds a sensible face-centred grid seed (3 × 3 × 3 grid with one vacancy).
2.  Lets every sphere radius and centre move under SLSQP while enforcing:
      • no overlaps      • stay entirely inside the cube
3.  Maximises Σ r  (implemented as minimising −Σ r).
"""

import numpy as np
from scipy.optimize import minimize


def _initial_layout(n: int = 26) -> tuple[np.ndarray, np.ndarray]:
    """Generate a near-regular seed layout and radii."""
    # 3×3×3 grid, centres at (k + 0.5) / 3 ∈ (0.167, 0.833)
    grid = (np.arange(3) + 0.5) / 3.0
    points = np.array([(x, y, z) for x in grid for y in grid for z in grid])

    # drop the geometrical centre → 26 points
    centre_idx = 13  # middle of the list, (0.5,0.5,0.5)
    centres = np.delete(points, centre_idx, axis=0)[:n]

    # conservative starting radius so neighbours don’t overlap
    base_r = 0.11
    radii = np.full(n, base_r)
    return centres, radii


def _objective(x: np.ndarray, n: int) -> float:
    """Return −Σ r (we minimise)."""
    return -x[3 * n :].sum()


def _constraints(x: np.ndarray, n: int) -> np.ndarray:
    """Inequality constraints g(x) ≥ 0  (non-overlap & inside cube)."""
    c = x[: 3 * n].reshape(n, 3)
    r = x[3 * n :]

    cons = []

    # pair-wise non-overlap:  dist − r_i − r_j ≥ 0
    for i in range(n):
        d = np.linalg.norm(c[i] - c[i + 1 :], axis=1)
        cons.extend(d - (r[i] + r[i + 1 :]))

    # stay inside cube faces
    cons.extend(c.flatten() - np.repeat(r, 3))  # each coord − r ≥ 0
    cons.extend(1.0 - c.flatten() - np.repeat(r, 3))  # 1 − coord − r ≥ 0
    return np.asarray(cons)


def construct_packing(n: int = 26):
    """
    Returns
    -------
    centres : (n,3) ndarray
    radii   : (n,) ndarray
    sum_r   : float
    """
    centres0, radii0 = _initial_layout(n)
    x0 = np.concatenate([centres0.ravel(), radii0])

    # variable bounds
    bounds = [(0.005, 0.995)] * (3 * n) + [(0.005, 0.5)] * n

    res = minimize(
        fun=_objective,
        x0=x0,
        args=(n,),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": _constraints, "args": (n,)},
        options={"maxiter": 600, "ftol": 1e-7, "disp": False},
    )

    # fall back to initial guess if optimisation failed
    best = res.x if res.success else x0
    centres = best[: 3 * n].reshape(n, 3)
    radii = best[3 * n :]
    radii = np.maximum(radii, 0.005)  # numerical safety

    return centres, radii, radii.sum()


# EVOLVE-BLOCK-END


def run_packing(n=26):
    """
    Run the sphere packing constructor for `n` unequal spheres in a unit cube.
    """
    centers, radii, sum_radii = construct_packing(n)
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the unequal sphere packing in a 3D unit cube.

    Args:
        centers: np.array of shape (n, 3) with (x, y, z) coordinates
        radii: np.array of shape (n) with radius of each sphere
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set limits to unit cube
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Draw spheres (represented as circles in 3D)
    for center, radius in zip(centers, radii):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(
            x, y, z, color=cm.viridis(np.random.rand()), rstride=4, cstride=4, alpha=0.6
        )

    # Set the title with the sum of radii
    ax.set_title(f"Unequal Sphere Packing (n={len(centers)}, sum of radii={np.sum(radii):.6f})")

    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
