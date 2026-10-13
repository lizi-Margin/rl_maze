import random
import numpy as np
from collections import defaultdict


class DoubleQLearningAgent:
    """
    Double Q-Learning agent (off-policy TD control).

    Vanilla Q-learning updates use max_a' Q(s',a') as the bootstrap target,
    which systematically overestimates action values because the maximum of
    noisy estimates is a biased (upward) estimator. Double Q-learning
    (van Hasselt, 2010) keeps two independent Q-tables and decouples action
    SELECTION from action EVALUATION:

        With 50% probability:
            A* = argmax_a' Q1(s',a')
            Q1(s,a) += α [r + γ Q2(s',A*) - Q1(s,a)]
        otherwise:
            A* = argmax_a' Q2(s',a')
            Q2(s,a) += α [r + γ Q1(s',A*) - Q2(s,a)]

    The behavior policy is ε-greedy w.r.t. the sum Q1+Q2.

    Note: epsilon is driven externally by the experiment harness
    (decayed once per episode).
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
                 epsilon_min=0.01):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

        self.q_table1 = defaultdict(lambda: np.zeros(4))
        self.q_table2 = defaultdict(lambda: np.zeros(4))

    def get_action(self, state, valid_actions):
        """Choose action using epsilon-greedy w.r.t. Q1 + Q2."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_sum = self.q_table1[state] + self.q_table2[state]
            valid_q = {action: q_sum[action] for action in valid_actions}
            return max(valid_q.keys(), key=lambda x: valid_q[x])

    def learn(self, state, action, reward, next_state, next_valid_actions):
        """
        Double Q-learning update: randomly pick one Q-table to update,
        and bootstrap from the OTHER table's value at the greedy action.
        """
        if random.uniform(0, 1) < 0.5:
            update_table, eval_table = self.q_table1, self.q_table2
        else:
            update_table, eval_table = self.q_table2, self.q_table1

        current_q = update_table[state][action]

        if next_valid_actions:
            next_q_values = {a: update_table[next_state][a] for a in next_valid_actions}
            best_next_action = max(next_q_values.keys(), key=lambda x: next_q_values[x])
            # Evaluate with the OTHER Q-table -> no maximisation bias
            next_q = eval_table[next_state][best_next_action]
        else:
            next_q = 0

        td_target = reward + self.gamma * next_q
        update_table[state][action] = current_q + self.lr * (td_target - current_q)

    def set_epsilon(self, epsilon):
        """Set exploration rate"""
        self.epsilon = max(epsilon, self.epsilon_min)

    def get_greedy_action(self, state, valid_actions):
        """Greedy action w.r.t. Q1+Q2 (for evaluation)."""
        q_sum = self.q_table1[state] + self.q_table2[state]
        valid_q = {action: q_sum[action] for action in valid_actions}
        return max(valid_q.keys(), key=lambda x: valid_q[x])
