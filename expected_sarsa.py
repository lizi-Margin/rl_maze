import random
import numpy as np
from collections import defaultdict


class ExpectedSARSAAgent:
    """
    Expected SARSA agent (on-policy TD control).

    Instead of sampling the next action A' like SARSA does, Expected SARSA
    uses the expectation over all next actions under the current policy:

        Q(s,a) = Q(s,a) + α * [r + γ * Σ_a' π(a'|s') Q(s',a') - Q(s,a)]

    where π is the current ε-greedy policy. This removes the variance
    introduced by sampling A', and typically converges faster and more
    stably than vanilla SARSA. (The lecture notes mention Expected Sarsa as
    an off-policy method that needs no importance sampling: if π is taken to
    be the greedy policy the update becomes Q-learning.)

    Note: epsilon is driven externally by the experiment harness
    (decayed once per episode) so that all algorithms are compared under
    identical exploration schedules.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
                 epsilon_min=0.01):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions: up, right, down, left

    def _epsilon_greedy_probs(self, state, valid_actions):
        """Probability of each valid action under the current ε-greedy policy."""
        q_values = self.q_table[state]
        n_valid = len(valid_actions)
        greedy_action = max(valid_actions, key=lambda a: q_values[a])

        probs = {}
        for a in valid_actions:
            probs[a] = self.epsilon / n_valid
            if a == greedy_action:
                probs[a] += 1.0 - self.epsilon
        return probs

    def get_action(self, state, valid_actions):
        """Choose action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            valid_q_values = {action: q_values[action] for action in valid_actions}
            return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])

    def learn(self, state, action, reward, next_state, next_valid_actions):
        """
        Update Q-table using the Expected SARSA update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_valid_actions: List of valid actions from next state
        """
        current_q = self.q_table[state][action]

        if next_valid_actions:
            # Expectation of Q(s', ·) under the current ε-greedy policy
            probs = self._epsilon_greedy_probs(next_state, next_valid_actions)
            expected_next_q = sum(probs[a] * self.q_table[next_state][a]
                                  for a in next_valid_actions)
        else:
            expected_next_q = 0

        td_target = reward + self.gamma * expected_next_q
        self.q_table[state][action] = current_q + self.lr * (td_target - current_q)

    def set_epsilon(self, epsilon):
        """Set exploration rate"""
        self.epsilon = max(epsilon, self.epsilon_min)

    def get_greedy_action(self, state, valid_actions):
        """Greedy action w.r.t. the current Q-table (for evaluation)."""
        q_values = self.q_table[state]
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])
