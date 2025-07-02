"""
Evaluator for the autocorrelation-constant (C₁ upper-bound) problem.

A higher score = tighter bound ⇒ score = -C₁_upper_bound
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Dict

import numpy as np


class TimeoutError(Exception):
    """Raised when the subprocess exceeds the allowed wall-clock time."""


def _run_with_timeout(program_path: str, timeout_s: int = 120) -> list[float]:
    """
    Execute `program_path` in a fresh Python interpreter, call
    `get_step_function_heights`, and return the resulting list of heights.

    All heavy work (search) happens inside the candidate program; we only
    collect the result.
    """
    # Temporary helper script
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as helper:
        helper.write(
            f"""
import importlib.util, os, pickle, sys, traceback

spec = importlib.util.spec_from_file_location("candidate", {program_path!r})
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

try:
    heights = candidate.get_step_function_heights()
    with open({helper.name!r} + ".res", "wb") as fh:
        pickle.dump({{"heights": heights}}, fh)
except Exception as e:
    with open({helper.name!r} + ".res", "wb") as fh:
        pickle.dump({{"error": str(e),
                      "trace": traceback.format_exc()}}, fh)
"""
        )
        helper_path = helper.name
    results_path = helper_path + ".res"

    try:
        proc = subprocess.Popen(
            [sys.executable, helper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"Candidate exceeded {timeout_s}s wall-clock limit.")

        if stdout:
            print(stdout.decode())
        if stderr:
            print(stderr.decode(), file=sys.stderr)

        if proc.returncode != 0:
            raise RuntimeError(f"Candidate exited with code {proc.returncode}")

        if not os.path.exists(results_path):
            raise RuntimeError("Result pickle missing – candidate likely crashed.")

        with open(results_path, "rb") as fh:
            res = pickle.load(fh)

        if "error" in res:
            raise RuntimeError(f"Candidate error: {res['error']}\n{res.get('trace', '')}")

        return res["heights"]

    finally:
        # Clean up helper files
        for p in (helper_path, results_path):
            if os.path.exists(p):
                os.remove(p)


def _c1_upper_bound(h: np.ndarray) -> float:
    n = h.size
    conv_max = np.convolve(h, h).max()
    return 2.0 * n * conv_max / h.sum() ** 2


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Full evaluation used by the optimiser.
    The *higher* the combined_score, the better (here: –C₁).
    """
    try:
        t0 = time.time()
        heights = _run_with_timeout(program_path, timeout_s=600)
        heights = np.asarray(heights, dtype=float)

        if not np.all(np.isfinite(heights)):
            raise ValueError("Non-finite values in heights.")

        c1 = _c1_upper_bound(heights)
        score = -float(c1)  # optimiser maximises

        elapsed = time.time() - t0
        print(f"Evaluation OK: C₁ ≤ {c1:.6f}, score = {score:.6f}, {elapsed:.2f}s elapsed")

        return {
            "C1_upper_bound": float(c1),
            "score": score,
            "combined_score": score,  # alias for consistency
            "eval_time": elapsed,
        }

    except TimeoutError as te:
        print(f"Timeout: {te}")
    except Exception as exc:
        print(f"Evaluation failed: {exc}\n{traceback.format_exc()}")

    # Failure fallback – worst possible score
    return {
        "C1_upper_bound": float("inf"),
        "score": -float("inf"),
        "combined_score": -float("inf"),
        "eval_time": 0.0,
    }


# Optional quick-check stage (shape/finite/timeout only)
def evaluate_stage1(program_path: str):
    try:
        heights = _run_with_timeout(program_path, timeout_s=120)
        heights = np.asarray(heights, dtype=float)
        valid = heights.ndim == 1 and np.all(np.isfinite(heights))
        return {"validity": 1.0 if valid else 0.0}
    except Exception:
        return {"validity": 0.0}


# Stage-2 simply delegates to full evaluation
def evaluate_stage2(program_path: str):
    return evaluate(program_path)
