"""
RL training server: this process now IS the training loop (Option A 
design,
decided this session -- see PROJECT_HANDOFF.md §5 for why a separate 
wrapper
process doesn't work with BookSim2's per-run socket connections).

For each episode:
  1. Pick an injection rate (sample_injection_rate).
  2. Launch BookSim2 as a subprocess with routing_function=rl at that 
rate.
  3. Accept its socket connection, serve decisions, buffer (state, 
action).
  4. When BookSim2 disconnects, wait for it to exit, parse its captured
     stdout for the final latency.
  5. Compute reward (compute_reward), call agent.remember() for every
     buffered transition, then agent.train_step().
  6. Loop to the next episode -- same agent object, weights persist.

STATUS: draft, written from the design discussion, NOT YET RUN. Needs a
real end-to-end smoke test (a few episodes) before trusting it. Known open
items marked TODO below -- see handoff doc §5/§Immediate next steps.
"""

import socket
import os
import re
import subprocess
import numpy as np

from rl_starter import DQNAgent, STATE_DIM, N_ACTIONS, N_DIRECTIONS
from reward import compute_reward, sample_injection_rate, NONCONVERGENCE_PENALTY

SOCKET_PATH = "/tmp/booksim_rl.sock"
MSG_DIM = STATE_DIM + N_DIRECTIONS  # 6 sensor values + 4 direction-validity flags

# Anchor everything to this file's location, not the cwd it was launched from.
RL_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(RL_DIR)
BOOKSIM_CWD = os.path.join(REPO_ROOT, "booksim2", "src")
BOOKSIM_BIN = os.path.join(BOOKSIM_CWD, "booksim")   # absolute -> cwd can't break it
BOOKSIM_CONFIG = "examples/mesh88_lat"               # stays relative; resolved inside BOOKSIM_CWD

if not os.path.isfile(BOOKSIM_BIN):
    raise SystemExit(f"booksim binary not found at {BOOKSIM_BIN} -- build it with `make` in booksim2/src")
if not os.path.isfile(os.path.join(BOOKSIM_CWD, BOOKSIM_CONFIG)):
    raise SystemExit(f"config not found at {os.path.join(BOOKSIM_CWD, BOOKSIM_CONFIG)}")


LATENCY_RE = re.compile(r"Packet latency average = ([0-9.]+) \(([0-9]+) samples\)")

N_EPISODES = 300  # TODO: bump way up once the smoke test passes


def run_one_episode(agent, rng, episode_num, greedy=False, forced_rate=None):
    injection_rate = forced_rate if forced_rate is not None else sample_injection_rate(rng)
    seed = episode_num  # reproducible per-episode traffic; TODO: decide if this is the right seeding strategy long-term

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    server.settimeout(30)  # TODO: pick a sane timeout -- what's reasonable for a converging vs. aborting run?

    cmd = [
        BOOKSIM_BIN, BOOKSIM_CONFIG,
        "routing_function=rl",
        "traffic=uniform",
        f"injection_rate={injection_rate}",
        "sim_count=3",
        f"seed={seed}",
    ]
    proc = subprocess.Popen(cmd, cwd=BOOKSIM_CWD, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)

    transitions = []  # list of (state, action)
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
                break  # BookSim2 disconnected

            line, buf = buf.split(b"\n", 1)
            parts = line.decode().strip().split()
            if len(parts) != MSG_DIM:
                continue

            values = [float(x) for x in parts]
            state = np.array(values[:STATE_DIM], dtype=np.float32)
            valid_mask = values[STATE_DIM:]
            valid_directions = [d for d, v in enumerate(valid_mask) if v > 0.5]
            if not valid_directions:
                # Shouldn't happen given min_adapt-style logic on the C++ side
                # (a router not at the destination always has >=1 productive
                # direction) -- fallback kept as a safety net, not expected
                # to trigger.
                valid_directions = list(range(N_DIRECTIONS))

            action = agent.act(state, valid_directions=valid_directions, greedy=greedy)
            conn.sendall(f"{action}\n".encode())
            transitions.append((state, action))
    finally:
        conn.close()

    stdout, _ = proc.communicate(timeout=30)  # TODO: what if BookSim2 hangs post-disconnect? pick a real timeout policy
    server.close()

    match = None
    for m in LATENCY_RE.finditer(stdout):
        match = m  # keep the last match -- the final converged line
    if match is None:
        # Never converged -- ran but hit latency_thres and aborted.
        episode_latency = None
        reward = NONCONVERGENCE_PENALTY
    else:
        episode_latency = float(match.group(1))
        mode_choices = [action // 4 for (_state, action) in transitions]
        reward = compute_reward(episode_latency, injection_rate, mode_choices)

    return episode_num, injection_rate, reward, len(transitions), stdout, transitions, episode_latency


def train_on_episode(agent, transitions, reward, n_train_steps=20):
    """Apply the single episode-level reward to every (state, action) pair
    buffered during the episode. s_next is approximated as the next
    buffered state in sequence (decisions interleave across many routers,
    so this is a simplification, not a per-router trajectory -- TODO:
    revisit whether this approximation is good enough, flag in report
    regardless)."""
    n = len(transitions)
    for i, (state, action) in enumerate(transitions):
        done = (i == n - 1)
        next_state = transitions[i + 1][0] if not done else state
        agent.remember(state, action, reward, next_state, done)
    for _ in range(n_train_steps):    
        agent.train_step()


def main():
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=1)
    rng = np.random.default_rng(seed=1)  # separate seed stream for rate sampling

    reward_history = []
    EVAL_RATES = [0.005, 0.010, 0.015, 0.018]  # fixed, representative, not random
    for ep in range(N_EPISODES):
        result = run_one_episode(agent, rng, ep)
        ep_num, rate, reward, n_decisions, stdout = result[:5]
        transitions = result[5] if len(result) > 5 else []
        latency = result[6] if len(result) > 6 else None

        if transitions:
            train_on_episode(agent, transitions, reward)
        agent.decay_epsilon()
        reward_history.append(reward)
        print(f"Episode {ep_num}: rate={rate:.4f} latency={latency} "
              f"decisions={n_decisions} reward={reward:.4f}")

        if ep % 25 == 0:
            for eval_rate in EVAL_RATES:
                eval_result = run_one_episode(agent, rng, ep, greedy=True, forced_rate=eval_rate)
                e_num, e_rate, e_reward, e_dec, e_stdout = eval_result[:5]
                e_lat = eval_result[6] if len(eval_result) > 6 else None
                print(f"  EVAL @ep{ep} rate={e_rate:.4f}: latency={e_lat} reward={e_reward:.4f}")

    print("Training loop finished.")
    print(f"First 5 rewards: {reward_history[:5]}")
    print(f"Last 5 rewards:  {reward_history[-5:]}")

if __name__ == "__main__":
    main()
