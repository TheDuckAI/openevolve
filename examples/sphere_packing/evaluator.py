import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(centers, radii):
    """
    Validate that spheres don't overlap and are inside the unit cube

    Args:
        centers: np.array of shape (n, 3) with (x, y, z) coordinates
        radii: np.array of shape (n) with radius of each sphere

    Returns:
        True if valid, False otherwise
    """
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(radii))):
        print("Non-finite coordinate or radius detected")
        return False

    if np.any(radii <= 0) or np.any(radii > 0.5):
        print("Invalid radius value detected")
        return False

    n = centers.shape[0]

    # Check if spheres are inside the unit cube
    for i in range(n):
        x, y, z = centers[i]
        r = radii[i]
        if (
            x - r < -1e-6
            or x + r > 1 + 1e-6
            or y - r < -1e-6
            or y + r > 1 + 1e-6
            or z - r < -1e-6
            or z + r > 1 + 1e-6
        ):
            print(f"Sphere {i} at ({x}, {y}, {z}) with radius {r} is outside the unit cube")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                print(f"Spheres {i} and {j} overlap: dist={dist}, r1+r2={radii[i] + radii[j]}")
                return False

    return True


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
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
    
    # Run the packing function
    print("Calling run_packing()...")
    centers, radii, sum_radii = program.run_packing()
    print(f"run_packing() returned successfully: sum_radii = {{sum_radii}}")

    # Save results to a file
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii
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

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["centers"], results["radii"], results["sum_radii"]
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
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        # Run the packing process
        centers, radii, sum_radii = run_with_timeout(program_path, timeout_seconds=600)

        # Validate the packing
        valid = validate_packing(centers, radii)

        # Shape validation for the packing
        shape_valid = centers.shape == (37, 3) and radii.shape == (37,)
        if not shape_valid:
            print(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}")
            valid = False

        # Calculate sum of radii
        sum_radii = np.sum(radii) if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Return evaluation metrics
        return {
            "sum_radii": float(sum_radii),
            "validity": float(validity),
            "combined_score": float(sum_radii),  # Just the sum of radii as the score
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return {
            "sum_radii": 0.0,
            "validity": 0.0,
            "combined_score": 0.0,
        }
