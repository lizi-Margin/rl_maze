import random
import numpy as np
from collections import defaultdict


class SarsaLambdaAgent:
    """
    Sarsa(λ): SARSA with eligibility traces (backward view, lecture 4.3).

    Sarsa(λ) unifies one-step SARSA (λ = 0) and Monte-Carlo-like updates
    (λ = 1) through the trace decay parameter λ. After each TD error

        δ = r + γ Q(s',a') - Q(s,a)

    the update is applied to the whole eligibility trace:

        Q(s,a)  <- Q(s,a) + α δ E(s,a)      for all (s,a)
        E(s,a)  <- γ λ E(s,a)               for all (s,a)
        E(S_t,A_t) <- E(S_t,A_t) + 1        (accumulating traces)

    'replacing' traces reset E(S_t,A_t) to 1 instead of adding 1, which
    behaves better with ε-greedy policies and is the default here.

    Because credit is propagated to the whole recent trajectory in a single
    episode, Sarsa(λ) usually learns considerably faster than one-step
    SARSA on problems where the reward is delayed (e.g. only at the goal).

    Note: epsilon is driven externally by the experiment harness
    (decayed once per episode). Traces are cleared at the start of
    every episode.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, lam=0.9,
                 epsilon=1.0, epsilon_min=0.01, trace_type='replacing'):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.lam = lam
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.trace_type = trace_type  # 'replacing' or 'accumulating'

        self.q_table = defaultdict(lambda: np.zeros(4))
        # Eligibility traces keyed by (state, action)
        self.traces = defaultdict(float)

    def start_episode(self):
        """Clear eligibility traces at the beginning of an episode."""
        self.traces.clear()

    def get_action(self, state, valid_actions):
        """Choose action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            valid_q_values = {action: q_values[action] for action in valid_actions}
            return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])

    def learn(self, state, action, reward, next_state, next_action):
        """
        Sarsa(λ) update using the backward view.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Action chosen in next state (None if terminal)
        """
        current_q = self.q_table[state][action]

        # Terminal state: no bootstrapping term
        next_q = self.q_table[next_state][next_action] if next_action is not None else 0.0
        delta = reward + self.gamma * next_q - current_q

        # Update trace of the current state-action pair
        key = (state, action)
        if self.trace_type == 'replacing':
            self.traces[key] = 1.0
        else:  # accumulating
            self.traces[key] += 1.0

        # Apply the TD error to every state-action pair in the trace
        for k, e in list(self.traces.items()):
            s, a = k
            self.q_table[s][a] += self.lr * delta * e
            # Decay all traces
            self.traces[k] = self.gamma * self.lam * e
            # Drop negligible traces to keep the dict small
            if self.traces[k] < 1e-6:
                del self.traces[k]

    def set_epsilon(self, epsilon):
        """Set exploration rate"""
        self.epsilon = max(epsilon, self.epsilon_min)

    def get_greedy_action(self, state, valid_actions):
        """Greedy action w.r.t. the current Q-table (for evaluation)."""
        q_values = self.q_table[state]
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values.keys(), key=lambda x: valid_q_values[x])
