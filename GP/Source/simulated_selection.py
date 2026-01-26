#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import inspect
import os
import sys
import time
from types import ModuleType
import numpy as np

# Best-effort per-call timeout using signal (Unix/POSIX only).
# On non-Unix platforms, we fall back to measuring elapsed time and failing if exceeded.
try:
    import signal
    _USE_SIGNAL_TIMEOUT = hasattr(signal, "SIGALRM")
except Exception:
    import types
    _USE_SIGNAL_TIMEOUT = False
    # Create a dummy signal module for type checker
    signal = types.SimpleNamespace(SIGALRM=None, signal=lambda *args: None, alarm=lambda *args: None)  # type: ignore

# derive a custom TimeoutError to avoid confusion with built-in TimeoutError
class TimeoutError_(Exception):
    pass

# Dummy handler for signal-based timeout
if _USE_SIGNAL_TIMEOUT:
    def _timeout_handler(_signum, _frame):
        raise TimeoutError_("selection() exceeded timeout")

# Load a Python module from a given file path
def load_module_from_path(py_path: str) -> ModuleType:
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"File not found: {py_path}")
    spec = importlib.util.spec_from_file_location("user_selector", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Retrieve the selection() function from the module, with basic validation
def get_selection_callable(mod: ModuleType):
    if not hasattr(mod, "selection"):
        raise AttributeError("selector.py does not define a function named 'selection'")
    fn = getattr(mod, "selection")
    if not callable(fn):
        raise TypeError("'selection' exists but is not callable")
    # Optional light signature sanity check: one positional parameter
    try:
        sig = inspect.signature(fn)
        if len(sig.parameters) < 1:
            raise TypeError("'selection' must accept at least one argument (fitnesses)")
    except Exception:
        # If signature introspection fails, proceed anyway since it's not critical
        pass
    return fn

# Generate synthetic fitness data based on absolute distance between y values abs(ground truth - predictions)
def generate_fitnesses(rng: np.random.Generator):
    """
    Create a list of N lists, each containing M floating-point numbers.
    N is number of individuals; M can represent number of objectives/metrics.

    Optimized: Uses numpy for vectorized random generation, then converts to list of lists.
    """
    # number of individual solutions list performance lists (predefined upper bound)
    N = 100
    # number of performance values per individual list (upper bound)
    M = 100

    # Generate all random values at once using numpy (much faster than nested list comprehension)
    fitness_array = rng.uniform(0.0, 10.0, size=(N, M))

    # Convert to list of lists as expected by selector functions
    fitnesses = fitness_array.tolist()

    # make sure fitness has the correct length
    assert len(fitnesses) == N
    # make sure each fitness entry has the correct length
    assert all(len(fit) == M for fit in fitnesses)
    # return stuff
    return fitnesses

# Write results to CSV file
def write_csv(out_path: str, success: bool, reason: str, avg_time_sec: float = 0.0, var_time_sec: float = 0.0):
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["success", "reason", "avg_time_sec", "var_time_sec"])
        writer.writerow([str(bool(success)), reason, avg_time_sec, var_time_sec])

# let it rip
def main():
    parser = argparse.ArgumentParser(
        description="Exercise selection() from a selector.py multiple times with validation."
    )
    parser.add_argument(
        "selector_path",
        help="Path to selector.py (e.g., Experiment_1/Rep_1/selector.py)",
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=100000,
        help="Number of calls to make (default: 100000)",
    )
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=10,
        help="Per-call timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for reproducibility (default: 1337)",
    )
    args = parser.parse_args()

    # Derive output CSV path from selector_path directory
    selector_dir = os.path.dirname(args.selector_path) or "."
    output_csv = os.path.join(selector_dir, "selection_test_result.csv")

    # initialize RNG for reproducible fitness data
    rng = np.random.default_rng(args.seed)

    # Pre-allocate numpy array for execution times (much faster than list appending)
    elapsed_times = np.zeros(args.calls, dtype=np.float64)

    # Load user module and selection()
    try:
        mod = load_module_from_path(args.selector_path)
        selection = get_selection_callable(mod)
    except Exception as e:
        write_csv(output_csv, False, f"Load error: {repr(e)}")
        print(f"[FAIL] {repr(e)}", file=sys.stderr)
        sys.exit(1)

    # Configure signal-based timeout if available
    if _USE_SIGNAL_TIMEOUT:
        signal.signal(signal.SIGALRM, _timeout_handler)

    # Calculate print frequency (print every 1% or every 1000 calls, whichever is larger)
    print_interval = max(1000, args.calls // 100)

    # N is constant (100 individuals), no need to extract it from generate_fitnesses every iteration
    N = 100

    # Execute up to args.calls times; stop on first unmet criterion
    for i in range(1, args.calls + 1):
        # Print progress at intervals to avoid I/O bottleneck
        if i % print_interval == 0 or i == 1 or i == args.calls:
            print(f"Call {i}/{args.calls}", flush=True)

        fitnesses = generate_fitnesses(rng)

        start = time.time()
        try:
            if _USE_SIGNAL_TIMEOUT:
                signal.alarm(args.timeout_sec)

            result = selection(fitnesses)

            if _USE_SIGNAL_TIMEOUT:
                signal.alarm(0)  # cancel alarm

        except TimeoutError_:
            reason = f"Timeout: selection() exceeded {args.timeout_sec}s on call {i}"
            write_csv(output_csv, False, reason)
            print(f"[FAIL] {reason}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            reason = f"Exception on call {i}: {repr(e)}"
            write_csv(output_csv, False, reason)
            print(f"[FAIL] {reason}", file=sys.stderr)
            sys.exit(1)

        # Compute elapsed time once after successful execution
        elapsed = time.time() - start

        # Check timeout for non-signal platforms
        if not _USE_SIGNAL_TIMEOUT and elapsed > args.timeout_sec:
            reason = f"Timeout: selection() exceeded {args.timeout_sec}s on call {i} (elapsed={elapsed:.2f}s)"
            write_csv(output_csv, False, reason)
            print(f"[FAIL] {reason}", file=sys.stderr)
            sys.exit(1)

        # Verify return type is a plain int (not bool) and within [0, N-1]
        # Combined validation for efficiency: check type and range in one go
        result_type = type(result)
        if isinstance(result, bool):
            reason = f"Invalid return type on call {i}: expected int index, got bool"
            write_csv(output_csv, False, reason)
            print(f"[FAIL] {reason}", file=sys.stderr)
            sys.exit(1)

        if not isinstance(result, (int, np.integer)):
            reason = f"Invalid return type on call {i}: expected int index (Python native or numpy), got {result_type.__name__}"
            write_csv(output_csv, False, reason)
            print(f"[FAIL] {reason}", file=sys.stderr)
            sys.exit(1)

        # Optimized range check: use result >= 0 and result < N (faster than chained comparison)
        if result < 0 or result >= N:
            reason = f"Out-of-range index on call {i}: got {result}, expected 0 <= index < {N}"
            write_csv(output_csv, False, reason)
            print(f"[FAIL] {reason}", file=sys.stderr)
            sys.exit(1)

        # elapsed time must be set here because no exception occurred
        assert elapsed is not None
        # Store elapsed time in pre-allocated array (faster than list append)
        elapsed_times[i - 1] = elapsed

    # make sure elapsed_times has correct length
    assert len(elapsed_times) == args.calls
    # If we get here, all calls satisfied the criteria
    write_csv(output_csv, True, "N/A", float(np.mean(elapsed_times)), float(np.var(elapsed_times)))
    print(f"[OK] All {args.calls} calls succeeded. Results written to {output_csv}")


if __name__ == "__main__":
    main()