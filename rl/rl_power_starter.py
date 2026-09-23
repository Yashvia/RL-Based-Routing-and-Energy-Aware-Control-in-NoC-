"""
Minimal power-only DQN agent: 3 actions (0=Active, 1=Balanced, 2=Eco), no
direction choice at all -- direction is fully determined by dor_next_mesh 
on
the C++ side (dor_rlpower_mesh). This is intentionally a SEPARATE, smaller
class from rl_starter.py's DQNAgent -- that class's act() hardcodes the
12-action (mode, direction) layout via module-level N_MODES/N_DIRECTIONS
constants and is not safe to reuse for a 3-action space. Reuses only the
validated QNetwork (forward/backward pass) from rl_starter.py.

Includes a target network from the start (the moving-target fix applied to
the full-RL agent this session) -- no reason to reintroduce that 
instability
here.
"""

import copy
import pickle
import random
from collections import deque

import numpy as np

from rl_starter import QNetwork  # reuse only the validated network code

STATE_DIM = 5   # own_occ, n_cong, s_cong, e_cong, w_cong
N_ACTIONS = 3    # 0=Active, 1=Balanced, 2=Eco


class PowerAgent:
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, gamma=0.9,
                 epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.97, seed=0):
        self.q_net = QNetwork(state_dim, n_actions, seed=seed)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_update_every = 150
        self.train_steps_count = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.replay = deque(maxlen=300_000)
        self.rng = random.Random(seed)
        self.best_eval_score = float("-inf")

    def act(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        q_values = self.q_net.forward(state)
        return int(np.argmax(q_values))

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
            td_error = float(np.clip(td_error, -5.0, 5.0))
            self.q_net.backward(a, td_error)

        self.train_steps_count += 1
        if self.train_steps_count % self.target_update_every == 0:
            self.target_net = copy.deepcopy(self.q_net)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"q_net": self.q_net, "best_eval_score": self.best_eval_score, "epsilon": self.epsilon}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_net = data["q_net"]
            self.target_net = copy.deepcopy(self.q_net)
            self.best_eval_score = data["best_eval_score"]
            self.epsilon = data.get("epsilon", self.epsilon)  # backward-compatible with older checkpoints
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
