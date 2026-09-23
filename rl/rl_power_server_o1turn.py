"""
Training server for the dor + RL-power agent (Tier 3, config 2 of 6).
Structurally mirrors rl_server.py's Option A episode loop, but simpler:
no direction masking, 3-action PowerAgent, separate socket so this can run
alongside the full-RL job on /tmp/booksim_rl.sock without colliding.

STATUS: draft, NOT YET RUN. Expect a debugging round, same as rl_mesh and
rl_server.py both needed on first run -- confirm dor_rlpower_mesh compiles
and is registered in gRoutingFunctionMap before running this.
"""

import os
import re
import socket
import subprocess

import numpy as np

from rl_power_starter import PowerAgent, STATE_DIM
from reward import compute_reward, sample_injection_rate_for_pattern, NONCONVERGENCE_PENALTY, sample_traffic_pattern
RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
BOOKSIM_CWD = os.path.join(REPO_ROOT, "booksim2", "src")
BOOKSIM_BIN = os.path.join(BOOKSIM_CWD, "booksim")
BOOKSIM_CONFIG = "examples/mesh88_lat"

# rl_power_server_o1turn.py
CHECKPOINT_PATH = "best_power_agent_o1turn.pkl"
agent = PowerAgent(seed=1)
if os.path.isfile(CHECKPOINT_PATH):
    agent.load(CHECKPOINT_PATH)
    print(f"Resumed from {CHECKPOINT_PATH}, best_eval_score={agent.best_eval_score:.4f}")
else:
    print("No checkpoint found, starting fresh.")

if not os.path.isfile(BOOKSIM_BIN):
    raise SystemExit(f"booksim binary not found at {BOOKSIM_BIN} -- build it with `make` in booksim2/src")
if not os.path.isfile(os.path.join(BOOKSIM_CWD, BOOKSIM_CONFIG)):
    raise SystemExit(f"config not found at {os.path.join(BOOKSIM_CWD, BOOKSIM_CONFIG)}")

SOCKET_PATH = "/tmp/booksim_rlpower_o1turn.sock"
MSG_DIM = STATE_DIM                          # 5 floats, no direction-validity flags

LATENCY_RE = re.compile(r"Packet latency average = ([0-9.]+) \(([0-9]+) samples\)")

N_EPISODES = 300

EVAL_RATES_BY_PATTERN = {
    "uniform":    [0.005, 0.010, 0.015, 0.018],
    "tornado":    [0.005, 0.008, 0.011, 0.013],
    "bitcomp":    [0.003, 0.006, 0.009, 0.0105],
    "transpose":  [0.002, 0.004, 0.006, 0.0068],
    "hotspot(0)": [0.0003, 0.0005, 0.0006, 0.0007],
}

EVAL_EVERY = 25
POWER_MW = {0: 226.99, 1: 152.63, 2: 111.09}  # Active, Balanced, Eco -- from Orion3 measurements, §2
ACTIVE_POWER_MW = POWER_MW[0]

def run_one_episode(agent, rng, episode_num, greedy=False, forced_rate=None, forced_traffic=None, force_active=False):
    traffic_pattern = forced_traffic if forced_traffic is not None else sample_traffic_pattern(rng)
    injection_rate = forced_rate if forced_rate is not None else sample_injection_rate_for_pattern(rng, traffic_pattern)
    seed = episode_num


    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    server.settimeout(30)  # TODO: same placeholder caveat as rl_server.py

    cmd = [
        BOOKSIM_BIN, BOOKSIM_CONFIG,
        "routing_function=o1turn_rlpower",
        f"traffic={traffic_pattern}",
        f"injection_rate={injection_rate}",
        "sim_count=3",
        f"seed={seed}",
    ]
    proc = subprocess.Popen(cmd, cwd=BOOKSIM_CWD, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)

    transitions = []
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

            state = np.array([float(x) for x in parts], dtype=np.float32)
            action = agent.act(state, greedy=greedy)
            conn.sendall(f"{action}\n".encode())
            transitions.append((state, action))
    finally:
        conn.close()

    stdout, _ = proc.communicate(timeout=30)
    server.close()

    match = None
    for m in LATENCY_RE.finditer(stdout):
        match = m
    if match is None:
        episode_latency = None
        reward = NONCONVERGENCE_PENALTY
        avg_power = None
        mode_counts = None
    else:
        episode_latency = float(match.group(1))
        mode_choices = [action for (_state, action) in transitions]  # already 0-2, no //4 needed
        reward = compute_reward(episode_latency, injection_rate, mode_choices, traffic_pattern)
        avg_power = sum(POWER_MW[m] for m in mode_choices) / len(mode_choices)
        mode_counts = {m: mode_choices.count(m) / len(mode_choices) for m in (0, 1, 2)}
    return episode_num, injection_rate, reward, len(transitions), stdout, transitions, episode_latency, avg_power, mode_counts


def train_on_episode(agent, transitions, reward, n_train_steps=20):
    n = len(transitions)
    for i, (state, action) in enumerate(transitions):
        done = (i == n - 1)
        next_state = transitions[i + 1][0] if not done else state
        agent.remember(state, action, reward, next_state, done)
    for _ in range(n_train_steps):
        agent.train_step()


def main():
    agent = PowerAgent(seed=1)
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
        mode_counts = result[8] if len(result) > 8 else None

        if transitions:
            train_on_episode(agent, transitions, reward)
        agent.decay_epsilon()
        reward_history.append(reward)
        EVAL_PATTERNS = ["uniform", "transpose", "hotspot(0)"] 
        if ep % EVAL_EVERY == 0:
            eval_scores = []
            for pattern in EVAL_PATTERNS:        
                for eval_rate in EVAL_RATES_BY_PATTERN[pattern]:
                    eval_result = run_one_episode(agent, rng, ep, greedy=True, forced_rate=eval_rate, forced_traffic=pattern)
                    e_num, e_rate, e_reward, e_dec, e_stdout = eval_result[:5]
                    e_lat = eval_result[6] if len(eval_result) > 6 else None
                    print(f"  EVAL @ep{ep} pattern={pattern} rate={e_rate:.4f}: latency={e_lat} reward={e_reward:.4f}")
                    eval_scores.append(e_reward)
            avg_eval = sum(eval_scores) / len(eval_scores)
            if avg_eval > agent.best_eval_score:
                agent.best_eval_score = avg_eval
                agent.save("best_power_agent_o1turn.pkl")
                print(f"  >> New best eval avg ({avg_eval:.4f}), saved to best_power_agent_o1turn.pkl")

        if avg_power is not None:
            savings_pct = 100 * (1 - avg_power / ACTIVE_POWER_MW)
            print(f"Episode {ep_num}: rate={rate:.4f} latency={latency} decisions={n_decisions} "
                  f"reward={reward:.4f} avg_power={avg_power:.1f}mW ({savings_pct:+.1f}% vs Active) "
                  f"modes={mode_counts}")
        else:
            print(f"Episode {ep_num}: rate={rate:.4f} latency={latency} decisions={n_decisions} reward={reward:.4f}")

    print("Training loop finished.")
    print(f"First 5 rewards: {reward_history[:5]}")
    print(f"Last 5 rewards:  {reward_history[-5:]}")


if __name__ == "__main__":
    main()
