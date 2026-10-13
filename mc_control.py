import random
import numpy as np
from collections import defaultdict


class OnPolicyMCControlAgent:
    """
    On-policy Monte-Carlo control (first-visit, GLIE).

    A direct policy-based counterpart of the TD control algorithms: instead
    of bootstrapping Q-values step by step, it waits until the end of an
    episode and updates Q(s,a) with the FULL sampled return

        Q(S_t,A_t) <- Q(S_t,A_t) + α [ G_t - Q(S_t,A_t) ],   G_t = Σ γ^k r_{t+k}

    Only the FIRST visit of each (s,a) pair in an episode is used
    (first-visit MC). The policy is improved after every episode to
    ε-greedy w.r.t. the current Q-table; with ε_k -> 0 (GLIE schedule,
    e.g. ε_k = 1/k driven by the harness) the algorithm converges to the
    optimal policy.

    MC control has higher variance than TD methods (returns depend on many
    random action choices) but no bias, and learns directly from complete
    episodes.

    Note: epsilon is driven externally by the experiment harness.
    """

    def __init__(self, learning_rate=0.1, discount_factor=1.0, epsilon=1.0,
                 epsilon_min=0.01, alpha_rule='constant'):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # 'constant': fixed α (non-stationary friendly, default)
        # 'average' : α = 1/N(s,a) sample averages
        self.alpha_rule = alpha_rule

        self.q_table = defaultdict(lambda: np.zeros(4))
        self.visit_counts = defaultdict(int)  # N(s,a) for sample averages
        self.episode_buffer = []  # list of (state, action, reward)

    def start_episode(self):
        """Begin collecting a new episode."""
        self.episode_buffer = []

    def get_action(self, state, valid_actions):
        """Choose action using epsilon-greedy w.r.t. the current Q-table."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            valid_q_values = {action: q_values[action] for action in valid_actions}
            return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])

    def learn(self, state, action, reward, done):
        """Buffer one transition; the actual update happens at episode end."""
        self.episode_buffer.append((state, action, reward))

    def end_episode(self):
        """First-visit MC update with the full sampled returns."""
        # Compute returns G_t backwards
        T = len(self.episode_buffer)
        returns = [0.0] * T
        g = 0.0
        for t in range(T - 1, -1, -1):
            g = self.episode_buffer[t][2] + self.gamma * g
            returns[t] = g

        visited = set()
        for t, (state, action, _r) in enumerate(self.episode_buffer):
            if (state, action) in visited:
                continue  # first-visit only
            visited.add((state, action))

            g_t = returns[t]
            key = (state, action)
            self.visit_counts[key] += 1

            if self.alpha_rule == 'average':
                alpha = 1.0 / self.visit_counts[key]
            else:
                alpha = self.lr

            self.q_table[state][action] += alpha * (g_t - self.q_table[state][action])

    def set_epsilon(self, epsilon):
        """Set exploration rate"""
        self.epsilon = max(epsilon, self.epsilon_min)

    def get_greedy_action(self, state, valid_actions):
        """Greedy action w.r.t. the current Q-table (for evaluation)."""
        q_values = self.q_table[state]
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])
