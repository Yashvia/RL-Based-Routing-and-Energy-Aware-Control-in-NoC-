"""
Unified evaluation: loads the best checkpoint from each trained config and
runs greedy (no-exploration) episodes across every rate x traffic pattern
combination. Reuses the actual run_one_episode functions from rl_server.py,
rl_power_server.py, rl_power_server_o1turn.py -- not reimplemented, so this
inherits whatever those scripts already do correctly.

This is EVALUATION ONLY -- no training happens here. Agents were trained on
uniform traffic only (per this week's scope decision); results on other
patterns test generalization, not in-distribution performance. Report this
honestly -- it's a legitimate finding either way.

Usage:
    python3 eval_comparison.py
Writes: rl_results.csv

Run this AFTER training has produced best_full_rl_agent.pkl,
best_power_agent.pkl, and best_power_agent_o1turn.pkl. Run baseline_sweep.py
separately (no checkpoints needed for that one).
"""

import csv
import os

import numpy as np

import rl_server
import rl_power_server
import rl_power_server_o1turn

from rl_starter import DQNAgent, STATE_DIM, N_ACTIONS
from rl_power_starter import PowerAgent
from reward import PATTERN_RATE_RANGES
RL_DIR = os.path.dirname(os.path.abspath(__file__))
def rates_for_pattern(pattern, n=6):
    lo, hi = PATTERN_RATE_RANGES[pattern]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]

PATTERNS = ["uniform", "transpose", "bitcomp", "tornado", "hotspot(0)"]


def eval_full_rl(checkpoint_path, force_active=False, config_name="full_rl"):
    if not os.path.isfile(checkpoint_path):
        print(f"  SKIP {config_name}: {checkpoint_path} not found")
        return []
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=0)
    agent.load(checkpoint_path)
    rng = np.random.default_rng(seed=0)
    rows = []
    for pattern in PATTERNS:
        for rate in rates_for_pattern(pattern):
            print(f"  {config_name}: pattern={pattern} rate={rate}")
            result = rl_server.run_one_episode(
                agent, rng, episode_num=0, greedy=True,
                forced_rate=rate, forced_traffic=pattern, force_active=force_active,
            )
            latency = result[6] if len(result) > 6 else None
            avg_power = result[7] if len(result) > 7 else None
            modes = result[8] if len(result) > 8 else None
            branches = result[9] if len(result) > 9 else None
            rows.append({
                "config": config_name, "pattern": pattern, "rate": rate,
                "latency": latency, "avg_power": avg_power,
                "modes": modes, "branch_dirs": branches,
            })
    return rows


def eval_power_only(module, checkpoint_path, config_name):
    if not os.path.isfile(checkpoint_path):
        print(f"  SKIP {config_name}: {checkpoint_path} not found")
        return []
    agent = PowerAgent(seed=0)
    agent.load(checkpoint_path)
    rng = np.random.default_rng(seed=0)
    rows = []
    for pattern in PATTERNS:
        for rate in rates_for_pattern(pattern):
            print(f"  {config_name}: pattern={pattern} rate={rate}")
            result = module.run_one_episode(
                agent, rng, episode_num=0, greedy=True,
                forced_rate=rate, forced_traffic=pattern,
            )
            latency = result[6] if len(result) > 6 else None
            avg_power = result[7] if len(result) > 7 else None
            modes = result[8] if len(result) > 8 else None
            rows.append({
                "config": config_name, "pattern": pattern, "rate": rate,
                "latency": latency, "avg_power": avg_power,
                "modes": modes, "branch_dirs": None,
            })
    return rows


def main():
    all_rows = []

    print("Evaluating full RL (routing + RL power)...")
    all_rows += eval_full_rl("best_full_rl_agent.pkl", force_active=False, config_name="RL_routing+RL_power")

    print("Evaluating full RL with power forced to Active (row 5, free)...")
    all_rows += eval_full_rl("best_full_rl_agent.pkl", force_active=True, config_name="RL_routing+Active")

    print("Evaluating dor + RL power...")
    all_rows += eval_power_only(rl_power_server, "best_power_agent.pkl", "dor+RL_power")

    print("Evaluating o1turn + RL power...")
    all_rows += eval_power_only(rl_power_server_o1turn, "best_power_agent_o1turn.pkl", "o1turn+RL_power")

    out_path = os.path.join(RL_DIR, "rl_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "pattern", "rate", "latency", "avg_power", "modes", "branch_dirs"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
