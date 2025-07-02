# EVOLVE-BLOCK-START
import numpy as np

BINS = 600  # number of equal-width sub-intervals on [-1/4, 1/4]
STEPS = 300_000  # hill-climb iterations (raise for a better bound)
DELTA = 1.0  # max mass moved in a single swap
SEED = 0  # RNG seed so runs are reproducible


def _c1_upper_bound(h: np.ndarray) -> float:
    """Compute the discrete upper bound for the current heights."""
    n = h.size
    conv_max = np.convolve(h, h).max()
    return 2.0 * n * conv_max / h.sum() ** 2


def _hill_climb() -> tuple[np.ndarray, float]:
    """Plain greedy search: keep a swap only if it improves C₁."""
    rng = np.random.default_rng(SEED)
    h = np.ones(BINS, dtype=float)  # start with the flat profile
    best_c = _c1_upper_bound(h)
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
        c = _c1_upper_bound(h)

        if c < best_c:  # accept improvement
            best_c, best_h = c, h.copy()
        else:  # undo swap
            h[i] -= delta
            h[j] += delta

    return best_h, best_c


def get_step_function_heights() -> list[float]:
    """Run the greedy search once and return the heights as a list."""
    heights, _ = _hill_climb()
    return heights.tolist()


# EVOLVE-BLOCK-END


def plot_step_function(step_heights_input: list[float], title=""):
    import matplotlib.pyplot as plt

    """Plots a step function with equally-spaced intervals on [-1/4,1/4]."""
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
    C_upper_bound = _c1_upper_bound(np.asarray(heights))

    print(f"Best step function found:  C₁ ≤ {C_upper_bound:.6f}")
    print("step_function_heights_1 = [")
    for k in range(0, len(heights), 10):
        print("    ", ", ".join(f"{v:.6f}" for v in heights[k : k + 10]), ",", sep="")
    print("]")

    plot_step_function(heights, title="Step function")
