"""
Full-RL training server (routing + power). This session's additions on top
of the previously-fixed version:
  - power/mode logging (avg_power, mode fractions) -- same as added to
    rl_power_server.py, so full-RL's energy behavior is finally visible
  - checkpointing on best eval score (best_full_rl_agent.pkl) -- protects
    against losing a good result the way ep75 was lost in the 300-episode
    run before this fix existed
  - branch-point direction tracking: whenever a router has exactly 2 valid
    productive directions (len(valid_directions) == 2 -- a genuine routing
    decision point, the same spot o1turn_mesh makes its fixed 50/50 XY/YX
    hash choice), log which one the agent actually picked. This is the
    only way to tell whether RL learned something structurally different
    from o1turn's fixed split, as opposed to just scoring similarly by
    luck -- reward alone can't distinguish those.

STATUS: draft, not yet run with these additions. Apply the same clipping
fix to DQNAgent (see rl_starter_DQNAgent_update.py) before running --
otherwise this will likely reproduce the ep75-then-collapse pattern from
the un-clipped 300-episode run.
"""

import os
import re
import socket
import subprocess

import numpy as np

from rl_starter import DQNAgent, STATE_DIM, N_ACTIONS, N_DIRECTIONS
from reward import compute_reward, sample_injection_rate_for_pattern, NONCONVERGENCE_PENALTY, sample_traffic_pattern

RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
BOOKSIM_CWD = os.path.join(REPO_ROOT, "booksim2", "src")
BOOKSIM_BIN = os.path.join(BOOKSIM_CWD, "booksim")
BOOKSIM_CONFIG = "examples/mesh88_lat"

# rl_server.py
CHECKPOINT_PATH = "best_full_rl_agent.pkl"
agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=1)
if os.path.isfile(CHECKPOINT_PATH):
    agent.load(CHECKPOINT_PATH)
    print(f"Resumed from {CHECKPOINT_PATH}, best_eval_score={agent.best_eval_score:.4f}")
else:
    print("No checkpoint found, starting fresh.")

if not os.path.isfile(BOOKSIM_BIN):
    raise SystemExit(f"booksim binary not found at {BOOKSIM_BIN} -- build it with `make` in booksim2/src")
if not os.path.isfile(os.path.join(BOOKSIM_CWD, BOOKSIM_CONFIG)):
    raise SystemExit(f"config not found at {os.path.join(BOOKSIM_CWD, BOOKSIM_CONFIG)}")

SOCKET_PATH = "/tmp/booksim_rl.sock"
MSG_DIM = STATE_DIM + N_DIRECTIONS  # 6 sensor values + 4 direction-validity flags

LATENCY_RE = re.compile(r"Packet latency average = ([0-9.]+) \(([0-9]+) samples\)")

POWER_MW = {0: 226.99, 1: 152.63, 2: 111.09}  # Active, Balanced, Eco -- Orion3 measurements, §2
ACTIVE_POWER_MW = POWER_MW[0]

N_EPISODES = 300

EVAL_RATES_BY_PATTERN = {
    "uniform":    [0.005, 0.010, 0.015, 0.018],
    "tornado":    [0.005, 0.008, 0.011, 0.013],
    "bitcomp":    [0.003, 0.006, 0.009, 0.0105],
    "transpose":  [0.002, 0.004, 0.006, 0.0068],
    "hotspot(0)": [0.0003, 0.0005, 0.0006, 0.0007],
}

EVAL_EVERY = 25


def run_one_episode(agent, rng, episode_num, greedy=False, forced_rate=None, forced_traffic=None, force_active=False):
    traffic_pattern = forced_traffic if forced_traffic is not None else sample_traffic_pattern(rng)
    injection_rate = forced_rate if forced_rate is not None else sample_injection_rate_for_pattern(rng, traffic_pattern)
    seed = episode_num

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    server.settimeout(30)

    cmd = [
        BOOKSIM_BIN, BOOKSIM_CONFIG,
        "routing_function=rl",
        f"traffic={traffic_pattern}",
        f"injection_rate={injection_rate}",
        "sim_count=3",
        f"seed={seed}",
    ]
    proc = subprocess.Popen(cmd, cwd=BOOKSIM_CWD, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)

    transitions = []
    branch_direction_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # NEW: counts at genuine 2-way branch points
    branch_point_total = 0                               # NEW

    try:
        conn, _ = server.accept()
    except socket.timeout:
        proc.kill()
        stdout, _ = proc.communicate()
        return episode_num, injection_rate, NONCONVERGENCE_PENALTY, 0, stdout

    buf = b""
    try:
        while True:
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
            if b"\n" not in buf:
                break

            line, buf = buf.split(b"\n", 1)
            parts = line.decode().strip().split()
            if len(parts) != MSG_DIM:
                continue

            values = [float(x) for x in parts]
            state = np.array(values[:STATE_DIM], dtype=np.float32)
            valid_mask = values[STATE_DIM:]
            valid_directions = [d for d, v in enumerate(valid_mask) if v > 0.5]
            if not valid_directions:
                valid_directions = list(range(N_DIRECTIONS))

            action = agent.act(state, valid_directions=valid_directions, greedy=greedy, force_active=force_active)
            conn.sendall(f"{action}\n".encode())
            transitions.append((state, action))

            if len(valid_directions) == 2:  # NEW: a genuine branch point, like o1turn's XY/YX choice
                direction = action % N_DIRECTIONS
                branch_direction_counts[direction] += 1
                branch_point_total += 1
    finally:
        conn.close()

    stdout, _ = proc.communicate(timeout=30)
    server.close()

    match = None
    for m in LATENCY_RE.finditer(stdout):
        match = m

    branch_summary = None
    if branch_point_total > 0:
        branch_summary = {d: round(c / branch_point_total, 3) for d, c in branch_direction_counts.items()}

    if match is None:
        episode_latency = None
        reward = NONCONVERGENCE_PENALTY
        avg_power = None
        mode_counts = None
    else:
        episode_latency = float(match.group(1))
        mode_choices = [action // N_DIRECTIONS for (_state, action) in transitions]
        reward = compute_reward(episode_latency, injection_rate, mode_choices, traffic_pattern)
        avg_power = sum(POWER_MW[m] for m in mode_choices) / len(mode_choices)
        mode_counts = {m: round(mode_choices.count(m) / len(mode_choices), 3) for m in (0, 1, 2)}

    return (episode_num, injection_rate, reward, len(transitions), stdout, transitions,
            episode_latency, avg_power, mode_counts, branch_summary)


def train_on_episode(agent, transitions, reward, n_train_steps=20):
    n = len(transitions)
    for i, (state, action) in enumerate(transitions):
        done = (i == n - 1)
        next_state = transitions[i + 1][0] if not done else state
        agent.remember(state, action, reward, next_state, done)
    for _ in range(n_train_steps):
        agent.train_step()


def _fmt_line(prefix, ep_num, rate, latency, reward, decisions=None, avg_power=None, modes=None, branches=None):
    parts = [f"{prefix} {ep_num}: rate={rate:.4f} latency={latency} reward={reward:.4f}"]
    if decisions is not None:
        parts.insert(1, f"decisions={decisions}")
    if avg_power is not None:
        savings = 100 * (1 - avg_power / ACTIVE_POWER_MW)
        parts.append(f"avg_power={avg_power:.1f}mW ({savings:+.1f}% vs Active)")
    if modes is not None:
        parts.append(f"modes={modes}")
    if branches is not None:
        parts.append(f"branch_dirs={branches}")
    return " ".join(parts)


def main():
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=1)
    if os.path.isfile(CHECKPOINT_PATH):
        agent.load(CHECKPOINT_PATH)
        print(f"Resumed from {CHECKPOINT_PATH}, best_eval_score={agent.best_eval_score:.4f}")
    else:
        print("No checkpoint found, starting fresh.")

    rng = np.random.default_rng(seed=1)

    reward_history = []

    for ep in range(N_EPISODES):
        result = run_one_episode(agent, rng, ep)
        ep_num, rate, reward, n_decisions, stdout = result[:5]
        transitions = result[5] if len(result) > 5 else []
        latency = result[6] if len(result) > 6 else None
        avg_power = result[7] if len(result) > 7 else None
        modes = result[8] if len(result) > 8 else None
        branches = result[9] if len(result) > 9 else None

        if transitions:
            train_on_episode(agent, transitions, reward)
        agent.decay_epsilon()
        reward_history.append(reward)

        print(_fmt_line("Episode", ep_num, rate, latency, reward, n_decisions, avg_power, modes, branches))
        EVAL_PATTERNS = ["uniform", "transpose", "hotspot(0)"]
        if ep % EVAL_EVERY == 0:
            eval_scores = []
            for pattern in EVAL_PATTERNS:
                for eval_rate in EVAL_RATES_BY_PATTERN[pattern]:
                    eval_result = run_one_episode(agent, rng, ep, greedy=True, forced_rate=eval_rate, forced_traffic=pattern)
                    e_num, e_rate, e_reward, e_dec, e_stdout = eval_result[:5]
                    e_lat = eval_result[6] if len(eval_result) > 6 else None
                    e_power = eval_result[7] if len(eval_result) > 7 else None
                    e_modes = eval_result[8] if len(eval_result) > 8 else None
                    e_branches = eval_result[9] if len(eval_result) > 9 else None
                    eval_line = f"  EVAL @ep{ep} pattern={pattern} rate={e_rate:.4f}: latency={e_lat} reward={e_reward:.4f}"
                    if e_power is not None:
                        savings = 100 * (1 - e_power / ACTIVE_POWER_MW)
                        eval_line += f" avg_power={e_power:.1f}mW ({savings:+.1f}% vs Active) modes={e_modes}"
                    if e_branches is not None:
                        eval_line += f" branch_dirs={e_branches}"
                    print(eval_line)
                    eval_scores.append(e_reward)

            avg_eval = sum(eval_scores) / len(eval_scores)
            if avg_eval > agent.best_eval_score:
                agent.best_eval_score = avg_eval
                agent.save("best_full_rl_agent.pkl")
                print(f"  >> New best eval avg ({avg_eval:.4f}), saved to best_full_rl_agent.pkl")

    print("Training loop finished.")
    print(f"First 5 rewards: {reward_history[:5]}")
    print(f"Last 5 rewards:  {reward_history[-5:]}")


if __name__ == "__main__":
    main()
