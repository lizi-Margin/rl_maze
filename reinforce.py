import random
import numpy as np
from collections import defaultdict


class ReinforceAgent:
    """
    REINFORCE with baseline (Monte-Carlo policy gradient).

    A policy-based method that learns a parametrized policy directly instead
    of a Q-table. The policy is a tabular softmax over action preferences
    H(s,a):

        π(a|s) = exp(H(s,a)) / Σ_a' exp(H(s,a'))

    After each episode, preferences are updated with the policy-gradient
    theorem using the FULL Monte-Carlo return (no bootstrapping):

        H(s_t,a) <- H(s_t,a) + α (G_t - b(s_t)) ∇ log π(a_t|s_t)[a]

    where ∇ log π(a_t|s_t)[a] = 1[a = a_t] - π(a|s_t), and the baseline
    b(s) is a Monte-Carlo estimate of the state value V(s) (running average
    of sampled returns). The baseline reduces the variance of the gradient
    without introducing bias. Set use_baseline=False to recover vanilla
    REINFORCE.

    This is the classic example of "policy-based RL + Monte-Carlo value
    estimation": both the policy AND the value baseline are learned from
    complete sampled episodes.

    Note: exploration comes from the softmax policy itself (no epsilon).
    """

    def __init__(self, learning_rate=0.01, discount_factor=0.99, use_baseline=True,
                 grid_size=5):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.use_baseline = use_baseline
        self.grid_size = grid_size

        # Action preferences (unnormalized log-policy)
        self.h_table = defaultdict(lambda: np.zeros(4))
        # Monte-Carlo baseline V(s) and visit counts N(s)
        self.v_table = defaultdict(float)
        self.visit_counts = defaultdict(int)

        self.episode_buffer = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _valid_actions(self, state):
        """Mirror of env.get_valid_actions (depends only on the state)."""
        row, col = state
        actions = []
        if row > 0:
            actions.append(0)
        if col < self.grid_size - 1:
            actions.append(1)
        if row < self.grid_size - 1:
            actions.append(2)
        if col > 0:
            actions.append(3)
        return actions if actions else [0, 1, 2, 3]

    def _softmax_probs(self, state, valid_actions):
        """π(a|s) restricted to the valid actions."""
        h = self.h_table[state]
        h_valid = np.array([h[a] for a in valid_actions])
        h_valid = h_valid - np.max(h_valid)  # numerical stability
        exp_h = np.exp(h_valid)
        return exp_h / np.sum(exp_h)

    # ------------------------------------------------------------------
    # agent interface
    # ------------------------------------------------------------------
    def start_episode(self):
        """Begin collecting a new episode."""
        self.episode_buffer = []

    def get_action(self, state, valid_actions):
        """Sample an action from the current softmax policy."""
        probs = self._softmax_probs(state, valid_actions)
        return random.choices(valid_actions, weights=probs, k=1)[0]

    def learn(self, state, action, reward, done):
        """Buffer one transition; the update happens at episode end."""
        self.episode_buffer.append((state, action, reward))

    def end_episode(self):
        """REINFORCE update with the MC return and MC baseline."""
        T = len(self.episode_buffer)
        returns = [0.0] * T
        g = 0.0
        for t in range(T - 1, -1, -1):
            g = self.episode_buffer[t][2] + self.gamma * g
            returns[t] = g

        for t, (state, action, _r) in enumerate(self.episode_buffer):
            valid_actions = self._valid_actions(state)
            probs = self._softmax_probs(state, valid_actions)
            g_t = returns[t]

            # Baseline: incremental MC average of returns for this state
            if self.use_baseline:
                advantage = g_t - self.v_table[state]
            else:
                advantage = g_t

            # Gradient ascent on E[return]: H(s,a) += α (G_t - b(s)) ∇logπ(a_t|s)
            # ∇logπ(a_t|s)[a] = 1[a = a_t] - π(a|s)
            for i, a in enumerate(valid_actions):
                grad_log_pi = (1.0 if a == action else 0.0) - probs[i]
                self.h_table[state][a] += self.lr * advantage * grad_log_pi

            # Update the baseline AFTER using it (every-visit MC average)
            if self.use_baseline:
                self.visit_counts[state] += 1
                self.v_table[state] += (g_t - self.v_table[state]) / self.visit_counts[state]

    def set_epsilon(self, epsilon):
        """No-op: exploration comes from the softmax policy itself."""
        pass

    def get_greedy_action(self, state, valid_actions):
        """Most probable action under the current policy (= target policy)."""
        probs = self._softmax_probs(state, valid_actions)
        return valid_actions[int(np.argmax(probs))]
