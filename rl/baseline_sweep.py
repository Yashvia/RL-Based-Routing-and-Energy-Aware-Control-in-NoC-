"""
Baseline sweep: dor and o1turn, fixed Active power (num_vcs=8), across every
rate x traffic pattern combination. No RL, no socket server -- just BookSim2
runs, parsed the same way as the original §1/§4b baseline table.

Usage:
    python3 baseline_sweep.py
Writes: baseline_results.csv
"""

import csv
import os
import re
import subprocess
from reward import PATTERN_RATE_RANGES

RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
BOOKSIM_CWD = os.path.join(REPO_ROOT, "booksim2", "src")
BOOKSIM_BIN = os.path.join(BOOKSIM_CWD, "booksim")
BOOKSIM_CONFIG = "examples/mesh88_lat"

LATENCY_RE = re.compile(r"Packet latency average = ([0-9.]+) \(([0-9]+) samples\)")

ROUTING_FUNCTIONS = ["dor", "o1turn"]
def rates_for_pattern(pattern, n=6):
    lo, hi = PATTERN_RATE_RANGES[pattern]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]
PATTERNS = ["uniform", "transpose", "bitcomp", "tornado", "hotspot(0)"]


def run_one(routing_function, rate, pattern, seed=1):
    cmd = [
        BOOKSIM_BIN, BOOKSIM_CONFIG,
        f"routing_function={routing_function}",
        f"traffic={pattern}",
        f"injection_rate={rate}",
        "sim_count=3",
        f"seed={seed}",
    ]
    result = subprocess.run(cmd, cwd=BOOKSIM_CWD, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=120)
    match = None
    for m in LATENCY_RE.finditer(result.stdout):
        match = m
    return float(match.group(1)) if match else None


def main():
    rows = []
    for rf in ROUTING_FUNCTIONS:
        for pattern in PATTERNS:
            for rate in rates_for_pattern(pattern):
                print(f"Running {rf}, {pattern}, rate={rate}...")
                latency = run_one(rf, rate, pattern)
                print(f"  -> latency={latency}")
                rows.append({
                    "config": f"{rf}+Active",
                    "pattern": pattern,
                    "rate": rate,
                    "latency": latency,
                })

    out_path = os.path.join(RL_DIR, "baseline_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "pattern", "rate", "latency"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
