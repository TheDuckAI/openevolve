"""
Evaluator for the second autocorrelation inequality (C₂ lower bound).

The optimisation objective is to **maximise** the returned `score`
which equals the computed lower bound R.
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
    """Raised when the subprocess exceeds the wall-clock limit."""


def _run_with_timeout(program_path: str, timeout_s: int = 120) -> list[float]:
    """
    Execute `program_path` in a clean interpreter, call
    `get_step_function_heights`, and return that list.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as helper:
        helper.write(
            f"""
import importlib.util, pickle, traceback, sys
spec = importlib.util.spec_from_file_location("cand", {program_path!r})
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

try:
    h = cand.get_step_function_heights()
    with open({helper.name!r} + ".res", "wb") as fh:
        pickle.dump({{"heights": h}}, fh)
except Exception as e:
    with open({helper.name!r} + ".res", "wb") as fh:
        pickle.dump({{"error": str(e), "trace": traceback.format_exc()}}, fh)
"""
        )
        helper_path = helper.name
    res_path = helper_path + ".res"

    try:
        proc = subprocess.Popen(
            [sys.executable, helper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"candidate exceeded {timeout_s}s limit")

        if out:
            print(out.decode())
        if err:
            print(err.decode(), file=sys.stderr)

        if proc.returncode != 0:
            raise RuntimeError(f"candidate exited with code {proc.returncode}")

        if not os.path.exists(res_path):
            raise RuntimeError("result pickle missing")

        with open(res_path, "rb") as fh:
            res = pickle.load(fh)

        if "error" in res:
            raise RuntimeError(f"candidate error: {res['error']}\n{res.get('trace', '')}")

        return res["heights"]

    finally:
        for p in (helper_path, res_path):
            if os.path.exists(p):
                os.remove(p)


def _c2_ratio(h: np.ndarray) -> float:
    conv = np.convolve(h, h)
    l1 = conv.sum()
    l_inf = conv.max()
    l2_sq = np.dot(conv, conv)
    return l2_sq / (l1 * l_inf)


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Main entry point for optimisers.
    Returns a dict with at least `score` and `combined_score`
    (both equal to the computed R).
    """
    try:
        t0 = time.time()
        heights = _run_with_timeout(program_path, timeout_s=600)
        h = np.asarray(heights, dtype=float)

        if h.ndim != 1 or not np.all(np.isfinite(h)) or h.size == 0:
            raise ValueError("invalid heights array")

        if h.sum() <= 0:
            raise ValueError("total mass must be positive")

        r_val = float(_c2_ratio(h))
        elapsed = time.time() - t0

        print(f"Evaluation OK: C₂ ≥ {r_val:.6f}  (elapsed {elapsed:.2f}s)")
        return {
            "C2_lower_bound": r_val,
            "score": r_val,
            "combined_score": r_val,
            "eval_time": elapsed,
        }

    except TimeoutError as te:
        print(f"Timeout: {te}")
    except Exception as exc:
        print(f"Evaluation failed: {exc}\n{traceback.format_exc()}")

    # Worst-case fallback
    return {
        "C2_lower_bound": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
        "eval_time": 0.0,
    }


def evaluate_stage1(program_path: str):
    """Quick validity/timeout check."""
    try:
        _ = _run_with_timeout(program_path, timeout_s=120)
        return {"validity": 1.0}
    except Exception:
        return {"validity": 0.0}


def evaluate_stage2(program_path: str):
    """Delegates to full evaluation."""
    return evaluate(program_path)
