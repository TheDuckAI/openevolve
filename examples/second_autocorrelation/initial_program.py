import numpy as np

BINS = 600  # number of bins on [−¼,¼]
STEPS = 300_000  # swap attempts
DELTA = 1.0  # maximum mass moved per accepted swap
SEED = 0  # RNG seed (reproducible runs)


def _c2_ratio(h: np.ndarray) -> float:
    """
    Compute   R = ‖f*f‖₂² / (‖f*f‖₁ · ‖f*f‖∞)   for a non-negative
    step function whose per-bin *masses* are given by h.
    """
    conv = np.convolve(h, h)
    l1 = conv.sum()
    l_inf = conv.max()
    l2_sq = np.dot(conv, conv)
    return l2_sq / (l1 * l_inf)


def _hill_climb() -> tuple[np.ndarray, float]:
    """Single greedy run; returns (best_heights, best_R)."""
    rng = np.random.default_rng(SEED)

    # uniform mass distribution to start, sums to 1
    h = np.full(BINS, 1.0 / BINS, dtype=float)
    best_r = _c2_ratio(h)
    best_h = h.copy()

    for _ in range(STEPS):
        i, j = rng.integers(0, BINS, size=2)
        if i == j:
            continue

        delta = (rng.random() - 0.5) * DELTA
        if h[i] + delta < 0 or h[j] - delta < 0:
            continue

        h[i] += delta
        h[j] -= delta
        r = _c2_ratio(h)

        if r > best_r:  # keep only if R increases
            best_r, best_h = r, h.copy()
        else:  # undo swap
            h[i] -= delta
            h[j] += delta

    return best_h, best_r


def get_step_function_heights() -> list[float]:
    """
    Run the greedy search once and return the bin masses as a list.
    """
    heights, _ = _hill_climb()
    return heights.tolist()


def plot_step_function(step_heights_input: list[float], title=""):
    """Plots a step function with equally-spaced intervals on [-1/4,1/4]."""
    import matplotlib.pyplot as plt

    num_steps = len(step_heights_input)

    # Generate x values for plotting (need to plot steps properly).
    step_edges_plot = np.linspace(-0.25, 0.25, num_steps + 1)
    x_plot = np.array([])
    y_plot = np.array([])

    for i in range(num_steps):
        x_start = step_edges_plot[i]
        x_end = step_edges_plot[i + 1]
        x_step_vals = np.linspace(x_start, x_end, 100)  # Points within each step.
        y_step_vals = np.full_like(x_step_vals, step_heights_input[i])
        x_plot = np.concatenate((x_plot, x_step_vals))
        y_plot = np.concatenate((y_plot, y_step_vals))

    # Plot the step function.
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.xlim([-0.3, 0.3])  # Adjust x-axis limits if needed.
    plt.ylim([-1, max(step_heights_input) * 1.2])  # Adjust y-axis limits.
    plt.grid(True)
    plt.step(
        step_edges_plot[:-1], step_heights_input, where="post", color="green", linewidth=2
    )  # Overlay with plt.step for clarity.
    plt.show()


if __name__ == "__main__":
    heights = get_step_function_heights()
    R = _c2_ratio(np.asarray(heights))

    print(f"Step function gives lower bound  C₂ ≥ {R:.6f}")
    print("step_function_heights_2 = [")
    for k in range(0, len(heights), 10):
        print("    ", ", ".join(f"{v:.6e}" for v in heights[k : k + 10]), ",", sep="")
    print("]")

    plot_step_function(heights, title="Discovered step function")
