# moving_sofa.py
"""
Simple benchmark script to search for a planar "moving sofa",
evaluate its validity through a unit-width right-angle hallway,
and render its motion as an animated GIF.

Dependencies
------------
    numpy
    shapely>=2.0   (geometry engine)
    matplotlib     (visualisation)
    imageio        (GIF writer – Pillow backend)

Install with, e.g.:
    pip install numpy shapely matplotlib imageio
"""

from __future__ import annotations

import pathlib
import random
from math import pi, sin

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon

# ----------------------------------------------------------------------------
# Global parameters (tweak these to trade speed for accuracy)
# ----------------------------------------------------------------------------
WALL_THICKNESS = 1.0  # corridor width (unit length)
N_ANGLES = 180  # samples from 0 → 90° (rotation grid)
N_SAMPLES = 120  # translations per angle sample

# VISUALIZATION PARAMETERS (reduced for memory efficiency)
VIZ_ANGLES = 36  # reduced animation frames (was 180)
VIZ_SAMPLES = 12  # reduced samples per angle (was 120)
GIF_FPS = 15  # reduced frame-rate (was 30)

# ----------------------------------------------------------------------------
# 1. Geometry helpers & validity test
# ----------------------------------------------------------------------------


def _is_pose_valid(p: Polygon) -> bool:
    """Return *False* if any vertex lies outside the L-shaped corridor."""
    xs, ys = p.exterior.coords.xy
    for x, y in zip(xs, ys):
        if x < 0 or y < 0:  # penetrates walls behind origin
            return False
        if x > WALL_THICKNESS and y > WALL_THICKNESS:
            return False  # enters forbidden square
    return True


def _motion_states(sofa: Polygon, n_angles: int = N_ANGLES, n_samples: int = N_SAMPLES):
    """Yield the sofa's successive poses along the classical pivot-and-slide path."""
    for i in range(n_angles + 1):
        theta = i * pi / (2 * n_angles)  # 0 → π/2  rad
        rot = affinity.rotate(sofa, theta * 180 / pi, origin=(0, 0), use_radians=False)
        r_max = 0.0 if theta == 0 else WALL_THICKNESS / sin(theta)
        for t in np.linspace(0.0, 1.0, n_samples):
            dx = r_max * (1.0 - t)
            dy = r_max * t
            yield affinity.translate(rot, dx, dy)


def evaluate_sofa(
    sofa: Polygon, *, n_angles: int = N_ANGLES, n_samples: int = N_SAMPLES
) -> tuple[float, bool]:
    """Return (area, validity) at the given sampling resolution."""
    for pose in _motion_states(sofa, n_angles, n_samples):
        if not _is_pose_valid(pose):
            return sofa.area, False
    return sofa.area, True


# ----------------------------------------------------------------------------
# 2. Shape encoding + stochastic search (greedy hill-climber)
# ----------------------------------------------------------------------------


def _convex_hull(points: np.ndarray) -> Polygon:
    """Closed convex polygon through *points*."""
    return MultiPoint(points).convex_hull


def _random_points(n: int, radius: float = 1.2) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.normal(0.0, radius, size=(n, 2))


def generate_sofa(n_ctrl: int = 8, iters: int = 4_000, step: float = 0.03) -> tuple[Polygon, float]:
    """Greedy hill-climb in control-point space.  Returns (polygon, area)."""
    # start from a unit square – guaranteed valid
    ctrl = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sofa = _convex_hull(ctrl)
    best_area, _ = evaluate_sofa(sofa)
    best_ctrl = ctrl.copy()

    for _ in range(iters):
        idx = random.randrange(len(ctrl))
        trial_ctrl = ctrl.copy()
        trial_ctrl[idx] += np.random.randn(2) * step
        trial_sofa = _convex_hull(trial_ctrl)
        area, ok = evaluate_sofa(trial_sofa)
        if ok and area > best_area:
            best_area = area
            best_ctrl = trial_ctrl
            ctrl = trial_ctrl
    return _convex_hull(best_ctrl), best_area


# ----------------------------------------------------------------------------
# 3. Memory-efficient visualizer
# ----------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")  # headless backend – prevents GUI windows
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    import imageio.v2 as imageio  # Pillow backend

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not found. GIF creation will use matplotlib's pillow writer.")


def visualize(
    sofa: Polygon, gif_path: str | pathlib.Path = "moving_sofa.gif", *, hallway_len: float = 6.0
):
    """Animate *sofa* moving through the corridor and save a GIF."""
    gif_path = pathlib.Path(gif_path)

    # static walls for rendering
    arm_x = Polygon([(0, 0), (hallway_len, 0), (hallway_len, WALL_THICKNESS), (0, WALL_THICKNESS)])
    arm_y = Polygon([(0, 0), (WALL_THICKNESS, 0), (WALL_THICKNESS, hallway_len), (0, hallway_len)])

    pad = 0.5
    xmin, ymin, xmax, ymax = -pad, -pad, hallway_len + pad, hallway_len + pad

    # Use reduced parameters for visualization
    states = list(_motion_states(sofa, VIZ_ANGLES, VIZ_SAMPLES))
    print(f"[visualize] Generating {len(states)} frames...")

    if HAS_IMAGEIO:
        # Create frames more efficiently with imageio
        frames = []
        fig, ax = plt.subplots(figsize=(4, 4))  # Smaller figure size
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

        for i, pose in enumerate(states):
            if i % 50 == 0:
                print(f"  Frame {i + 1}/{len(states)}")

            ax.clear()
            ax.set_aspect("equal")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.axis("off")

            # Draw walls
            ax.fill(*arm_x.exterior.xy, color="#e0e0e0")
            ax.fill(*arm_y.exterior.xy, color="#e0e0e0")

            # Draw sofa
            ax.fill(*pose.exterior.xy, color="#2d6cdf")

            # Convert to image
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(buf)

        plt.close(fig)

        # Write GIF
        print("[visualize] Writing GIF...")
        imageio.mimsave(gif_path, frames, fps=GIF_FPS)
        print(f"[visualize] GIF written → {gif_path}  ({len(states)} frames)")
    else:
        # Use matplotlib animation instead
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")
        ax.fill(*arm_x.exterior.xy, color="#e0e0e0")
        ax.fill(*arm_y.exterior.xy, color="#e0e0e0")
        (sofa_patch,) = ax.fill([], [], color="#2d6cdf")

        def _update(pose):
            sofa_patch.set_xy(np.asarray(pose.exterior.coords))
            return (sofa_patch,)

        ani = animation.FuncAnimation(
            fig, _update, frames=states, interval=1000 / GIF_FPS, blit=True
        )

        # Try to save with pillow writer
        try:
            ani.save(gif_path, writer="pillow", fps=GIF_FPS)
            print(f"[visualize] GIF written → {gif_path}  ({len(states)} frames)")
        except Exception as e:
            print(f"[visualize] Failed to save GIF: {e}")
            print("Try installing imageio with: pip install imageio")

        plt.close(fig)


def visualize_static(
    sofa: Polygon,
    image_path: str | pathlib.Path = "moving_sofa_static.png",
    *,
    hallway_len: float = 6.0,
):
    """Create a static visualization showing multiple sofa positions."""
    image_path = pathlib.Path(image_path)

    # static walls for rendering
    arm_x = Polygon([(0, 0), (hallway_len, 0), (hallway_len, WALL_THICKNESS), (0, WALL_THICKNESS)])
    arm_y = Polygon([(0, 0), (WALL_THICKNESS, 0), (WALL_THICKNESS, hallway_len), (0, hallway_len)])

    pad = 0.5
    xmin, ymin, xmax, ymax = -pad, -pad, hallway_len + pad, hallway_len + pad

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    # Draw walls
    ax.fill(*arm_x.exterior.xy, color="#e0e0e0")
    ax.fill(*arm_y.exterior.xy, color="#e0e0e0")

    # Show several key positions
    key_states = list(_motion_states(sofa, 8, 4))  # Just 9 angles, 4 samples each
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(key_states)))

    for i, (pose, color) in enumerate(zip(key_states, colors)):
        alpha = 0.3 if i < len(key_states) - 1 else 0.8  # Highlight final position
        ax.fill(*pose.exterior.xy, color=color, alpha=alpha)

    plt.title(f"Moving Sofa (Area ≈ {sofa.area:.4f})")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize_static] Static image written → {image_path}")


# ----------------------------------------------------------------------------
# 4. Command-line entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    sofa_poly, area = generate_sofa()
    print(f"Found sofa with area ≈ {area:.4f} (sampling grid {N_ANGLES}×{N_SAMPLES})")

    # Create static visualization first (safer)
    visualize_static(sofa_poly)

    # Ask before creating GIF
    create_gif = input("Create animated GIF? (y/n): ").strip().lower()
    if create_gif in ["y", "yes"]:
        visualize(sofa_poly)
        print("Done – open moving_sofa.gif to watch the journey!")
    else:
        print("Skipping GIF creation. Check moving_sofa_static.png for the result!")
