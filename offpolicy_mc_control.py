import random
import numpy as np
from collections import defaultdict


class OffPolicyMCControlAgent:
    """
    Off-policy Monte-Carlo control with weighted importance sampling.

    The behavior policy b (used to act, driven by the harness with a FIXED
    exploration rate) collects episodes, while we learn the action values of
    a GREEDY-CENTERED target policy π that is ε_π-soft:

        π(a|s) = 1-ε_π+ε_π/n  for the greedy action
        π(a|s) = ε_π/n        otherwise

    Because π ≠ b, the returns are corrected with importance-sampling
    ratios

        W_t = Π_{k=t}^{T-1} π(A_k|S_k) / b(A_k|S_k)

    and Q is updated with the WEIGHTED importance-sampling estimator:

        C(s,a) <- C(s,a) + W
        Q(s,a) <- Q(s,a) + (W / C(s,a)) [ G_t - Q(s,a) ]

    Weighted IS has much lower variance than ordinary IS. We keep the
    target slightly soft (ε_π = 0.05 by default): with a fully greedy
    target, W_t becomes 0 for every trajectory that ever deviates from π
    and learning is cripplingly slow - with a soft target all trajectories
    keep a positive weight and the correction stays effective. Behavior
    uses ε_b = 0.1 > ε_π, so π is (much) more greedy than b.

    Note: the behavior epsilon is kept FIXED by the experiment harness so
    that exploration never dies.
    """

    def __init__(self, learning_rate=0.1, discount_factor=1.0, epsilon=0.1,
                 epsilon_min=0.001, grid_size=5, target_epsilon=0.05):
        self.lr = learning_rate
        self.gamma = discount_factor
        # epsilon of the BEHAVIOR policy b
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.grid_size = grid_size
        # epsilon of the TARGET policy π (kept soft so WIS keeps signal)
        self.target_epsilon = target_epsilon

        self.q_table = defaultdict(lambda: np.zeros(4))
        self.c_weights = defaultdict(float)  # C(s,a): cumulative IS weights
        self.episode_buffer = []  # list of (state, action, reward)

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

    def _greedy_action(self, state):
        """Greedy action of the target policy π under the current Q-table."""
        q_values = self.q_table[state]
        return max(self._valid_actions(state), key=lambda a: q_values[a])

    def _target_probs(self, state):
        """π(a|s) of the ε_π-soft greedy-centered target policy."""
        valid_actions = self._valid_actions(state)
        greedy = self._greedy_action(state)
        n = len(valid_actions)
        probs = {}
        for a in valid_actions:
            probs[a] = self.target_epsilon / n
            if a == greedy:
                probs[a] += 1.0 - self.target_epsilon
        return probs

    def _behavior_prob(self, state, action):
        """b(a|s) of the ε-greedy behavior policy (must match get_action)."""
        valid_actions = self._valid_actions(state)
        greedy = self._greedy_action(state)
        n = len(valid_actions)
        if action == greedy:
            return 1.0 - self.epsilon + self.epsilon / n
        return self.epsilon / n

    # ------------------------------------------------------------------
    # agent interface
    # ------------------------------------------------------------------
    def start_episode(self):
        """Begin collecting a new episode."""
        self.episode_buffer = []

    def get_action(self, state, valid_actions):
        """Behavior policy b: epsilon-greedy w.r.t. the current Q-table."""
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
        """First-visit off-policy MC update (weighted importance sampling)."""
        T = len(self.episode_buffer)
        returns = [0.0] * T
        g = 0.0
        for t in range(T - 1, -1, -1):
            g = self.episode_buffer[t][2] + self.gamma * g
            returns[t] = g

        visited = set()
        W = 1.0
        for t in range(T - 1, -1, -1):
            state, action, _r = self.episode_buffer[t]

            # Importance-sampling ratio π(A_t|S_t)/b(A_t|S_t) with the soft
            # target π: strictly positive, so every trajectory contributes,
            # but trajectories that stay greedy accumulate larger W.
            W *= self._target_probs(state)[action] / self._behavior_prob(state, action)

            key = (state, action)
            if key not in visited and W > 0.0:
                visited.add(key)
                self.c_weights[key] += W
                alpha_w = W / self.c_weights[key]
                self.q_table[state][action] += alpha_w * (returns[t] - self.q_table[state][action])

    def set_epsilon(self, epsilon):
        """Set exploration rate of the BEHAVIOR policy."""
        self.epsilon = max(epsilon, self.epsilon_min)

    def get_greedy_action(self, state, valid_actions):
        """Greedy action w.r.t. the current Q-table (= target policy π)."""
        q_values = self.q_table[state]
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])
