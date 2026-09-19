"""
RL starter scaffold for the NoC routing + energy-mode project.

WHY A MOCK ENVIRONMENT FIRST:
Wiring a live Python agent directly into BookSim2's C++ simulation loop
(so it makes a real decision at every packet, every cycle) is real
engineering work -- BookSim2 would need to pause and call out to Python
at every routing decision. Rather than debug the DQN algorithm itself
*and* a brand-new C++/Python bridge at the same time, this file gives
you a fast, simplified mock of a router's environment so you can build,
train, and sanity-check the agent in minutes. Swap MockNoCEnv for a real
BookSim2-backed environment later (Sprint 4 in your timeline) once this
part is working -- the DQN/agent code doesn't need to change, only
the environment does.

STATE (per routing decision, at one router):
  [local_buffer_occupancy,      # 0-1, fraction full
   north_congestion, south_congestion, east_congestion, west_congestion,  # 0-1 each
   current_mode]                # 0=Active, 1=Balanced, 2=Eco

ACTION (one combined choice per decision):
  direction in {N, S, E, W}  x  mode in {Active, Balanced, Eco}
  = 4 x 3 = 12 discrete actions, encoded as a single integer 0-11.

REWARD:
  -(latency_cost + energy_cost + wakeup_penalty_if_mode_changed)
  i.e. the agent is penalized for both delay and energy; the balance
  between them is controlled by ENERGY_WEIGHT below -- tune this once
  real BookSim2/Orion3 numbers are flowing in.
"""

import numpy as np
import random
import copy
from collections import deque

# ---------------------------------------------------------------------
# Config -- tune these against your real Orion3/BookSim2 numbers later
# ---------------------------------------------------------------------
N_DIRECTIONS = 4          # N, S, E, W
N_MODES = 3               # Active, Balanced, Eco
N_ACTIONS = N_DIRECTIONS * N_MODES
STATE_DIM = 6              # see docstring above

ENERGY_WEIGHT = 0.5         # how much energy cost matters vs latency
WAKEUP_PENALTY = {0: 0.0, 1: 0.2, 2: 0.8}   # Active/Balanced/Eco, from your Orion3 table


def decode_action(a):
    """Map a single action integer back to (direction, mode)."""
    direction = a % N_DIRECTIONS
    mode = a // N_DIRECTIONS
    return direction, mode


# ---------------------------------------------------------------------
# Mock environment -- stand-in for BookSim2. Replace internals with
# real simulator calls once the agent is validated here.
# ---------------------------------------------------------------------
class MockNoCEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.buffer_occ = self.rng.uniform(0, 1)
        self.neighbor_congestion = self.rng.uniform(0, 1, size=4)
        self.mode = 0  # start Active
        return self._state()

    def _state(self):
        return np.array(
            [self.buffer_occ, *self.neighbor_congestion, self.mode],
            dtype=np.float32,
        )

    def step(self, action):
        direction, new_mode = decode_action(action)

        # --- Mock physics: replace this block with real BookSim2/Orion3
        # feedback once integrated. For now it's a plausible stand-in so
        # the training loop has something non-trivial to learn from.
        congestion_on_chosen_dir = self.neighbor_congestion[direction]
        latency_cost = congestion_on_chosen_dir * (1.0 + 0.5 * self.buffer_occ)

        mode_energy = {0: 1.0, 1: 0.65, 2: 0.45}[new_mode]  # relative, from your Orion3 totals
        energy_cost = mode_energy

        wakeup_cost = WAKEUP_PENALTY[new_mode] if new_mode != self.mode else 0.0

        reward = -(latency_cost + ENERGY_WEIGHT * energy_cost + wakeup_cost)

        # advance mock world state
        self.mode = new_mode
        self.buffer_occ = np.clip(
            self.buffer_occ + self.rng.normal(0, 0.1) - 0.05 * (1 - congestion_on_chosen_dir),
            0, 1,
        )
        self.neighbor_congestion = np.clip(
            self.neighbor_congestion + self.rng.normal(0, 0.1, size=4), 0, 1
        )
        # --- end mock physics block

        done = False  # episodic structure TBD once real traffic patterns are wired in
        return self._state(), reward, done, {}


# ---------------------------------------------------------------------
# Minimal DQN, pure NumPy (no torch/tensorflow dependency needed to
# get started -- swap in a real framework once you're ready to scale
# up network size or use a GPU).
# ---------------------------------------------------------------------
class QNetwork:
    def __init__(self, state_dim, n_actions, hidden=32, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0, 0.1, (state_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, (hidden, n_actions))
        self.b2 = np.zeros(n_actions)

    def save(self, path):
        np.savez(path, **{k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)})

    def load(self, path):
        data = np.load(path)
        for k in data.files:
            setattr(self, k, data[k])
    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)  # ReLU
        self.q = self.h1 @ self.W2 + self.b2
        return self.q

    def backward(self, action, td_error):
        # gradient of MSE loss w.r.t. the one Q-value we updated
        dq = np.zeros_like(self.q)
        dq[action] = -2 * td_error

        dW2 = np.outer(self.h1, dq)
        db2 = dq
        dh1 = dq @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)
        dW1 = np.outer(self.x, dz1)
        db1 = dz1

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


class DQNAgent:
    def __init__(self, state_dim, n_actions, gamma=0.9, epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.97, seed=0):
        self.q_net = QNetwork(state_dim, n_actions, seed=seed)
        self.target_net = copy.deepcopy(self.q_net)   # NEW: frozen copy for stable targets
        self.target_update_every = 50                  # NEW: how often to refresh it (in train_step calls)
        self.train_steps_count = 0       
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.replay = deque(maxlen=300_000)
        self.rng = random.Random(seed)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        self.q_net.save(path)

    def load(self, path):
        self.q_net.load(path)

    def act(self, state, valid_directions=None, greedy=False):
        if valid_directions is None:
            valid_directions = list(range(N_DIRECTIONS))
        valid_actions = [mode * N_DIRECTIONS + d for mode in range(N_MODES) for d in valid_directions]
        if (not greedy) and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        q_values = self.q_net.forward(state)
        masked_q = np.full_like(q_values, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]
        return int(np.argmax(masked_q))

    def remember(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.replay) < 32:
            return
        batch = self.rng.sample(self.replay, 32)
        for s, a, r, s_next, done in batch:
            q_next = self.target_net.forward(s_next)
            target = r if done else r + self.gamma * np.max(q_next)
            q_current = self.q_net.forward(s)
            td_error = target - q_current[a]
            self.q_net.backward(a, td_error)
        self.train_steps_count += 1
        if self.train_steps_count % self.target_update_every == 0:
            self.target_net = copy.deepcopy(self.q_net)   # NEW: periodically sync target to live weights


# ---------------------------------------------------------------------
# Training loop -- run this file directly to sanity-check the whole
# pipeline end-to-end against the mock environment.
# ---------------------------------------------------------------------
def train(n_episodes=200, steps_per_episode=50):
    env = MockNoCEnv(seed=1)
    agent = DQNAgent(STATE_DIM, N_ACTIONS, seed=1)

    episode_rewards = []
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        agent.epsilon = max(0.05, agent.epsilon * 0.99)  # decay exploration

        if (ep + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"Episode {ep+1:4d}  avg reward (last 20) = {avg:.3f}  epsilon = {agent.epsilon:.3f}")

    return episode_rewards


if __name__ == "__main__":
    rewards = train()
    print("\nTraining complete against the mock environment.")
    print(f"First 20-episode avg reward: {np.mean(rewards[:20]):.3f}")
    print(f"Last 20-episode avg reward:  {np.mean(rewards[-20:]):.3f}")
    print("\nIf the last average is clearly better than the first, the")
    print("agent is learning something from this mock world -- the")
    print("plumbing (state/action/reward/replay/training) works.")
    print("Next real step: replace MockNoCEnv with a wrapper around")
    print("actual BookSim2 output (per-router buffer occupancy, neighbor")
    print("congestion from print_activity, and Orion3-derived energy")
    print("costs per mode) -- the agent code above does not need to change.")
