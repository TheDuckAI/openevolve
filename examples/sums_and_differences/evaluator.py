"""
Evaluator for sums and differences of finite sets problem (B.6)
Finds sets U that improve the lower bound for constant C_6
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def compute_lower_bound(u):
    """
    Returns the lower bound obtained from the input set u, which must satisfy min(u) == 0.

    Args:
        u: list or array of non-negative integers containing 0

    Returns:
        Lower bound for C_6 using the formula:
        C_6 >= 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    u = np.array(u, dtype=int)

    if len(u) == 0:
        return 0.0

    if np.min(u) != 0:
        print(f"Set U must contain 0 and be non-negative; got minimum value {np.min(u)}")
        return 0.0

    if np.any(u < 0):
        print("Set U must be non-negative; got negative values")
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

    # Check constraint |U - U| <= 2*max(U) + 1
    if u_minus_u_size > 2 * max_u + 1:
        print(f"Constraint |U - U| <= 2*max(U) + 1 violated: {u_minus_u_size} > {2 * max_u + 1}")
        return 0.0

    if u_plus_u_size == 0:
        return 0.0

    # Compute the lower bound
    if u_minus_u_size <= u_plus_u_size:
        return 0.0  # Invalid - we need |U-U| > |U+U| for a meaningful bound

    bound = 1.0 + np.log(u_minus_u_size / u_plus_u_size) / np.log(2 * max_u + 1)

    return bound


def validate_set(u):
    """
    Validate that the set U satisfies all constraints

    Args:
        u: list or array of integers

    Returns:
        tuple (is_valid, error_message)
    """
    if not isinstance(u, (list, np.ndarray)):
        return False, "U must be a list or array"

    u = np.array(u, dtype=int)

    if len(u) == 0:
        return False, "U cannot be empty"

    if len(u) > 100000:  # Reasonable upper limit for computation
        return False, f"U is too large ({len(u)} elements), maximum 100000"

    if np.min(u) != 0:
        return False, f"U must contain 0, got minimum {np.min(u)}"

    if np.any(u < 0):
        return False, "U must contain only non-negative integers"

    # Check for duplicates
    if len(set(u)) != len(u):
        return False, "U must not contain duplicates"

    max_u = int(np.max(u))

    # Compute set sizes for constraint checking
    u_minus_u = set()
    for i in u:
        for j in u:
            u_minus_u.add(i - j)

    if len(u_minus_u) > 2 * max_u + 1:
        return (
            False,
            f"Constraint |U - U| <= 2*max(U) + 1 violated: {len(u_minus_u)} > {2 * max_u + 1}",
        )

    return True, ""


def run_with_timeout(program_path, timeout_seconds=60):
    """
    Run the program in a separate process with timeout

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        u, bound tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the construction function
    print("Calling construct_set()...")
    u, bound = program.construct_set()
    print(f"construct_set() returned successfully: bound = {{bound}}")

    # Save results to a file
    results = {{
        'u': u,
        'bound': bound
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["u"], results["bound"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the program by running it and checking the lower bound

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        start_time = time.time()

        # Run the program with timeout
        u, reported_bound = run_with_timeout(program_path, timeout_seconds=300)

        end_time = time.time()
        eval_time = end_time - start_time

        # Validate the set
        is_valid, error_msg = validate_set(u)
        if not is_valid:
            print(f"Invalid set: {error_msg}")
            return {
                "bound": 0.0,
                "validity": 0.0,
                "set_size": 0,
                "eval_time": float(eval_time),
                "combined_score": 0.0,
            }

        # Compute the actual bound
        actual_bound = compute_lower_bound(u)

        # Check if reported bound matches computed bound
        if abs(actual_bound - reported_bound) > 1e-6:
            print(
                f"Warning: Reported bound {reported_bound} doesn't match computed bound {actual_bound}"
            )

        # Validity score
        validity = 1.0 if is_valid else 0.0

        # Combined score - simply the bound value (higher is better)
        # Scale by validity to ensure invalid solutions get 0 score
        combined_score = actual_bound * validity

        print(
            f"Evaluation: valid={is_valid}, bound={actual_bound:.6f}, "
            f"set_size={len(u)}, time={eval_time:.2f}s"
        )

        return {
            "bound": float(actual_bound),
            "validity": float(validity),
            "set_size": int(len(u)),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "bound": 0.0,
            "validity": 0.0,
            "set_size": 0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }


def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
    """
    try:
        try:
            u, reported_bound = run_with_timeout(program_path, timeout_seconds=120)

            # Quick validation
            is_valid, error_msg = validate_set(u)
            if not is_valid:
                print(f"Stage 1: Invalid set - {error_msg}")
                return {"validity": 0.0, "combined_score": 0.0, "error": error_msg}

            # Compute bound
            actual_bound = compute_lower_bound(u)

            # Combined score is simply the bound value
            combined_score = actual_bound if is_valid else 0.0

            return {
                "validity": 1.0 if is_valid else 0.0,
                "bound": float(actual_bound),
                "combined_score": float(combined_score),
            }

        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {"validity": 0.0, "combined_score": 0.0, "error": "Timeout"}
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}

    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {e}")
        return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation
    """
    return evaluate(program_path)
