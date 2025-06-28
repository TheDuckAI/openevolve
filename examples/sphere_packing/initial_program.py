# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each sphere in a 3D unit cube.
    The radii are limited by the distance to the cube borders and other spheres.

    Args:
        centers: np.array of shape (n, 3) with (x, y, z) coordinates

    Returns:
        np.array of shape (n) with radius of each sphere
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit by distance to cube borders
    for i in range(n):
        x, y, z = centers[i]
        radii[i] = min(x, y, z, 1 - x, 1 - y, 1 - z)

    # Limit by distance to other spheres
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])

            # If the current radii would cause overlap
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


def construct_packing(n=26):
    """
    Construct a specific arrangement of `n` unequal spheres in a 1x1x1 unit cube
    that attempts to maximize the sum of their radii.

    Args:
        n: Number of spheres

    Returns:
        centers, radii, sum_of_radii
        centers: np.array of shape (n, 3) with (x, y, z) coordinates
        radii: np.array of shape (n) with radius of each sphere
        sum_of_radii: Sum of all radii
    """
    # Initialize random centers for spheres within the unit cube
    centers = np.random.rand(n, 3)

    # Compute radii for each sphere
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


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
