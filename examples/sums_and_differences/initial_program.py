# EVOLVE-BLOCK-START
"""Constructor-based set construction for sums and differences problem (B.6)"""

import numpy as np


def construct_set():
    """
    Construct a specific set U of non-negative integers containing 0
    that attempts to maximize the lower bound for C_6.

    The bound is: C_6 >= 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)

    Returns:
        Tuple of (u, bound)
        u: list of non-negative integers containing 0
        bound: computed lower bound for C_6
    """

    # Start with a simple construction based on powers of 2
    # This is known to give some improvement over the baseline

    # Strategy: Use a geometric progression to create large difference sets
    # while keeping sum sets relatively small

    # Base construction: powers of 2 up to some limit
    max_power = 10  # Will give us numbers up to 2^10 = 1024

    u = [0]  # Must contain 0

    # Add powers of 2
    for i in range(1, max_power + 1):
        u.append(2**i)

    # Add some intermediate values to increase |U-U| while controlling |U+U|
    # Add numbers of the form 2^i + 2^j for small i, j
    for i in range(3):
        for j in range(i + 1, 4):
            val = 2**i + 2**j
            if val not in u and val <= 1024:
                u.append(val)

    # Add a few more strategic values
    # Numbers that create many differences but fewer sums
    additional_values = [3, 5, 7, 13, 17, 31]
    for val in additional_values:
        if val not in u:
            u.append(val)

    # Sort the set
    u.sort()

    # Compute the bound
    bound = compute_lower_bound_internal(u)

    return u, bound


def compute_lower_bound_internal(u):
    """
    Internal function to compute the lower bound for C_6

    Args:
        u: list of non-negative integers containing 0

    Returns:
        Lower bound value
    """
    u = np.array(u, dtype=int)

    if len(u) == 0 or np.min(u) != 0:
        return 0.0

    max_u = int(np.max(u))

    # Compute U - U (difference set)
    u_minus_u = set()
    for i in u:
        for j in u:
            u_minus_u.add(i - j)

    # Compute U + U (sum set)
    u_plus_u = set()
    for i in u:
        for j in u:
            u_plus_u.add(i + j)

    u_minus_u_size = len(u_minus_u)
    u_plus_u_size = len(u_plus_u)

    # Check constraint
    if u_minus_u_size > 2 * max_u + 1:
        return 0.0

    if u_plus_u_size == 0 or u_minus_u_size <= u_plus_u_size:
        return 0.0

    # Compute the bound
    bound = 1.0 + np.log(u_minus_u_size / u_plus_u_size) / np.log(2 * max_u + 1)

    return bound


def analyze_construction(u):
    """
    Analyze the properties of a construction

    Args:
        u: list of integers

    Returns:
        Dictionary with analysis results
    """
    u = np.array(u, dtype=int)
    max_u = int(np.max(u))

    # Compute sets
    u_minus_u = set()
    u_plus_u = set()

    for i in u:
        for j in u:
            u_minus_u.add(i - j)
            u_plus_u.add(i + j)

    return {
        "set_size": len(u),
        "max_value": max_u,
        "difference_set_size": len(u_minus_u),
        "sum_set_size": len(u_plus_u),
        "ratio": len(u_minus_u) / len(u_plus_u) if len(u_plus_u) > 0 else 0,
        "constraint_bound": 2 * max_u + 1,
        "constraint_satisfied": len(u_minus_u) <= 2 * max_u + 1,
    }


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_construction():
    """Run the set construction"""
    u, bound = construct_set()
    return u, bound


def visualize_construction(u):
    """
    Visualize the construction and its properties

    Args:
        u: list of integers representing the set U
    """
    analysis = analyze_construction(u)
    bound = compute_lower_bound_internal(u)

    print(f"Set U: {u}")
    print(f"Set size: {analysis['set_size']}")
    print(f"Max value: {analysis['max_value']}")
    print(f"|U - U|: {analysis['difference_set_size']}")
    print(f"|U + U|: {analysis['sum_set_size']}")
    print(f"Ratio |U-U|/|U+U|: {analysis['ratio']:.4f}")
    print(f"Constraint bound (2*max+1): {analysis['constraint_bound']}")
    print(f"Constraint satisfied: {analysis['constraint_satisfied']}")
    print(f"Lower bound for C_6: {bound:.6f}")

    # Compare to known bounds
    baseline = 1.14465
    alphaevolve1 = 1.1479
    alphaevolve2 = 1.1584

    print("\nComparison:")
    print(f"Baseline (Gyarmati et al.): {baseline:.6f}")
    print(f"AlphaEvolve result 1: {alphaevolve1:.6f}")
    print(f"AlphaEvolve result 2: {alphaevolve2:.6f}")
    print(f"Our result: {bound:.6f}")

    if bound > alphaevolve2:
        print("🎉 NEW RECORD! Better than AlphaEvolve!")
    elif bound > alphaevolve1:
        print("🎯 Excellent! Better than first AlphaEvolve result!")
    elif bound > baseline:
        print("✅ Good! Better than baseline!")
    else:
        print("❌ Needs improvement")


if __name__ == "__main__":
    u, bound = run_construction()
    print(f"Constructed set with lower bound: {bound:.6f}")

    # Uncomment to see detailed analysis:
    visualize_construction(u)
