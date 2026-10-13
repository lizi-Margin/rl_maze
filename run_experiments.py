"""
run_experiments.py — Unified evaluation harness for tabular RL algorithms.

All 8 agents are trained on the SAME SimpleMaze environment under IDENTICAL
conditions: same seeds, same max-step limit, and (for epsilon-greedy
methods) the same per-episode epsilon schedule. This keeps the comparison
fair — differences in learning curves can then be attributed to the
algorithms themselves.

Experiments
-----------
main          : all 8 algorithms, main comparison (multi-seed)
alpha         : learning-rate sensitivity for Q-Learning / SARSA / Expected SARSA
epsilon       : exploration-schedule sensitivity for Q-Learning / SARSA
lambda        : trace-decay sensitivity for Sarsa(lambda)
reinforce     : REINFORCE with vs without baseline
overestimation: max-Q along the optimal path: Q-Learning vs Double Q-Learning

Usage
-----
    python run_experiments.py --experiment all
    python run_experiments.py --experiment main --episodes 2000 --seeds 20
"""

import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import SimpleMaze
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from expected_sarsa import ExpectedSARSAAgent
from double_q_learning import DoubleQLearningAgent
from sarsa_lambda import SarsaLambdaAgent
from mc_control import OnPolicyMCControlAgent
from offpolicy_mc_control import OffPolicyMCControlAgent
from reinforce import ReinforceAgent

# Chinese font for figure labels (fallbacks keep it portable)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

MAX_STEPS = 100          # episode step cap, same as train.py
OPTIMAL_STEPS = 7        # length of the shortest safe path (BFS, see below)
EVAL_EPISODES = 20       # greedy evaluation episodes after training

# ----------------------------------------------------------------------
# Algorithm registry
# ----------------------------------------------------------------------
# mode: how the harness feeds experiences to the agent
#   'q'        -> learn(s, a, r, s', valid_actions(s'))   (TD, bootstrap max/expectation)
#   'sarsa'    -> next action chosen first, learn(s, a, r, s', a')
#   'episodic' -> learn(s, a, r, done) buffers; end_episode() does the update
# eps: per-episode epsilon schedule applied by the harness
#   ('exp', eps0, decay, eps_min) / ('fixed', eps) / ('glie',) / None (softmax policy)
EPS_TD = ('exp', 1.0, 0.98, 0.01)

ALGO_SPECS = {
    'q_learning': dict(
        title='Q-Learning', category='value', mode='q', eps=EPS_TD,
        color='#1f77b4',
        ctor=lambda **kw: QLearningAgent(**{**dict(
            learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
            epsilon_decay=1.0, epsilon_min=0.01), **kw})),
    'sarsa': dict(
        title='SARSA', category='value', mode='sarsa', eps=EPS_TD,
        color='#ff7f0e',
        ctor=lambda **kw: SARSAAgent(**{**dict(
            learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
            epsilon_decay=1.0, epsilon_min=0.01), **kw})),
    'expected_sarsa': dict(
        title='Expected SARSA', category='value', mode='q', eps=EPS_TD,
        color='#2ca02c',
        ctor=lambda **kw: ExpectedSARSAAgent(**{**dict(
            learning_rate=0.1, discount_factor=0.95, epsilon=1.0), **kw})),
    'double_q_learning': dict(
        title='Double Q-Learning', category='value', mode='q', eps=EPS_TD,
        color='#d62728',
        ctor=lambda **kw: DoubleQLearningAgent(**{**dict(
            learning_rate=0.1, discount_factor=0.95, epsilon=1.0), **kw})),
    'sarsa_lambda': dict(
        title='Sarsa(λ=0.9)', category='value', mode='sarsa', eps=EPS_TD,
        color='#9467bd',
        ctor=lambda **kw: SarsaLambdaAgent(**{**dict(
            learning_rate=0.1, discount_factor=0.95, lam=0.9, epsilon=1.0), **kw})),
    'mc_control': dict(
        title='MC Control (on-policy)', category='policy', mode='episodic',
        eps=('exp', 1.0, 0.995, 0.05), color='#8c564b',
        ctor=lambda **kw: OnPolicyMCControlAgent(**{**dict(
            learning_rate=0.1, discount_factor=1.0, epsilon=1.0,
            epsilon_min=0.05, alpha_rule='constant'), **kw})),
    'offpolicy_mc_control': dict(
        title='MC Control (off-policy WIS)', category='policy', mode='episodic',
        eps=('fixed', 0.4), color='#e377c2',
        ctor=lambda **kw: OffPolicyMCControlAgent(**{**dict(
            learning_rate=0.1, discount_factor=1.0, epsilon=0.4,
            target_epsilon=0.15), **kw})),
    'reinforce': dict(
        title='REINFORCE + baseline', category='policy', mode='episodic',
        eps=None, color='#7f7f7f',
        ctor=lambda **kw: ReinforceAgent(**{**dict(
            learning_rate=0.005, discount_factor=0.99, use_baseline=True), **kw})),
}

MAIN_ORDER = ['q_learning', 'sarsa', 'expected_sarsa', 'double_q_learning',
              'sarsa_lambda', 'mc_control', 'offpolicy_mc_control', 'reinforce']


def make_agent(algo_name, overrides=None):
    spec = ALGO_SPECS[algo_name]
    return spec['ctor'](**(overrides or {}))


# ----------------------------------------------------------------------
# Environment analysis helpers (ground truth for the fixed environment)
# ----------------------------------------------------------------------
def move(env, state, action):
    """Deterministic transition of SimpleMaze (position only)."""
    row, col = state
    if action == 0:
        return (max(0, row - 1), col)
    if action == 1:
        return (row, min(env.size - 1, col + 1))
    if action == 2:
        return (min(env.size - 1, row + 1), col)
    return (row, max(0, col - 1))


def reward_done(env, pos):
    """Reward/termination when landing on pos."""
    if pos in env.traps:
        return -50, True
    if pos == env.target_pos:
        return 100, True
    return -1, False


def optimal_path(env):
    """BFS shortest path over safe cells (avoids real traps)."""
    start, goal = env.start_pos, env.target_pos
    prev = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s == goal:
            break
        for a in env.get_valid_actions(s):
            ns = move(env, s, a)
            if ns in env.traps or ns in prev:
                continue
            prev[ns] = (s, a)
            q.append(ns)
    path, s = [], goal
    while s is not None:
        path.append(s)
        s = prev[s][0] if prev[s] is not None else None
    return path[::-1]  # start -> goal


def value_iteration(env, gamma, theta=1e-12):
    """Exact optimal state values V*(s) of the (known, deterministic) MDP."""
    safe = [(r, c) for r in range(env.size) for c in range(env.size)
            if (r, c) not in env.traps and (r, c) != env.target_pos]
    V = {s: 0.0 for s in safe}
    while True:
        delta = 0.0
        for s in safe:
            best = max(reward_done(env, move(env, s, a))[0] +
                       (0 if reward_done(env, move(env, s, a))[1] else gamma * V[move(env, s, a)])
                       for a in env.get_valid_actions(s))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            return V


def path_max_q(agent, env, path_states):
    """Mean over path states of max_a Q(s,a) (for the overestimation study)."""
    vals = []
    for s in path_states:
        if s == env.target_pos:
            continue
        va = env.get_valid_actions(s)
        if hasattr(agent, 'q_table1'):  # Double Q: use (Q1+Q2)/2
            q = (agent.q_table1[s] + agent.q_table2[s]) / 2.0
        elif hasattr(agent, 'q_table'):
            q = agent.q_table[s]
        else:
            return np.nan
        vals.append(max(q[a] for a in va))
    return float(np.mean(vals)) if vals else np.nan


# ----------------------------------------------------------------------
# Training / evaluation loops
# ----------------------------------------------------------------------
def apply_eps_schedule(agent, ep, schedule):
    kind = schedule[0]
    if kind == 'exp':
        _, eps0, decay, eps_min = schedule
        agent.set_epsilon(max(eps_min, eps0 * (decay ** ep)))
    elif kind == 'fixed':
        agent.set_epsilon(schedule[1])
    elif kind == 'glie':
        agent.set_epsilon(max(1.0 / (ep + 1), 0.001))
    else:
        raise ValueError(schedule)


def train_one(algo_name, episodes, seed, overrides=None, eps_schedule=None,
              max_steps=MAX_STEPS, probe_every=0, probe_path=None):
    """Train one agent for `episodes` episodes. Returns per-episode metrics."""
    spec = ALGO_SPECS[algo_name]
    random.seed(seed)
    np.random.seed(seed)
    env = SimpleMaze()
    agent = make_agent(algo_name, overrides)
    schedule = eps_schedule if eps_schedule is not None else spec['eps']
    mode = spec['mode']

    rewards = np.zeros(episodes)
    steps = np.zeros(episodes)
    success = np.zeros(episodes)
    probes = []

    start_ep = getattr(agent, 'start_episode', None)
    end_ep = getattr(agent, 'end_episode', None)

    for ep in range(episodes):
        if schedule is not None:
            apply_eps_schedule(agent, ep, schedule)
        if start_ep:
            start_ep()

        state = env.reset()
        total_r, n, done, last_r = 0.0, 0, False, 0.0

        if mode == 'sarsa':  # choose first action before the loop
            action = agent.get_action(state, env.get_valid_actions(state))

        while not done and n < max_steps:
            if mode == 'q':
                valid_actions = env.get_valid_actions(state)
                action = agent.get_action(state, valid_actions)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state,
                            env.get_valid_actions(next_state))
            elif mode == 'sarsa':
                next_state, reward, done = env.step(action)
                next_valid = env.get_valid_actions(next_state)
                next_action = None if done else agent.get_action(next_state, next_valid)
                agent.learn(state, action, reward, next_state, next_action)
                action = next_action
            elif mode == 'episodic':
                valid_actions = env.get_valid_actions(state)
                action = agent.get_action(state, valid_actions)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, done)
            else:
                raise ValueError(mode)

            state = next_state
            total_r += reward
            last_r = reward
            n += 1

        if end_ep:
            end_ep()

        rewards[ep] = total_r
        steps[ep] = n
        success[ep] = 1.0 if (done and last_r > 0) else 0.0

        if probe_every and ep % probe_every == 0 and probe_path is not None:
            probes.append(path_max_q(agent, env, probe_path))

    eval_res = evaluate_greedy(agent, env)
    result = dict(rewards=rewards, steps=steps, success=success,
                  eval=eval_res, probes=np.array(probes) if probes else None)
    return result, agent


def evaluate_greedy(agent, env, n_episodes=EVAL_EPISODES, max_steps=MAX_STEPS):
    """Run the learned greedy policy; report success rate / steps / return."""
    greedy_fn = getattr(agent, 'get_greedy_action', None)
    eps_backup = getattr(agent, 'epsilon', None)
    if greedy_fn is None and hasattr(agent, 'set_epsilon'):
        agent.set_epsilon(0.0)

    succ, step_list, ret_list, paths = [], [], [], []
    for _ in range(n_episodes):
        state = env.reset()
        done, n, total, last_r = False, 0, 0.0, 0.0
        path = [state]
        while not done and n < max_steps:
            va = env.get_valid_actions(state)
            action = greedy_fn(state, va) if greedy_fn else agent.get_action(state, va)
            state, reward, done = env.step(action)
            path.append(state)
            total += reward
            last_r = reward
            n += 1
        ok = 1.0 if (done and last_r > 0) else 0.0
        succ.append(ok)
        step_list.append(n)
        ret_list.append(total)
        paths.append(path)

    if greedy_fn is None and eps_backup is not None:
        agent.set_epsilon(eps_backup)

    ok_arr = np.array(succ) > 0.5
    succ_steps = np.array(step_list)[ok_arr]
    return dict(success=float(np.mean(succ)),
                avg_steps=float(np.mean(step_list)),
                avg_steps_success=float(np.mean(succ_steps)) if len(succ_steps) else np.nan,
                avg_return=float(np.mean(ret_list)),
                optimal=float(np.mean([1.0 if n == OPTIMAL_STEPS else 0.0
                                       for n in succ_steps])) if len(succ_steps) else 0.0,
                paths=paths)


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------
def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def episodes_to_success(success, thr=0.9, w=100):
    """First episode index (1-based) whose w-window success MA reaches thr."""
    if len(success) < w:
        return np.nan
    ma = moving_average(success, w)
    idx = np.where(ma >= thr)[0]
    return int(idx[0] + w) if len(idx) else np.nan  # episode count up to that point


def summarize(runs):
    """Aggregate a list of run dicts into mean/std metrics."""
    rewards = np.array([r['rewards'] for r in runs])       # (seeds, episodes)
    success = np.array([r['success'] for r in runs])
    steps = np.array([r['steps'] for r in runs])
    return dict(
        rewards_mean=rewards.mean(0), rewards_std=rewards.std(0),
        success_mean=success.mean(0), success_std=success.std(0),
        steps_mean=steps.mean(0), steps_std=steps.std(0),
        final_reward=rewards[:, -100:].mean(),
        final_reward_std=rewards[:, -100:].mean(1).std(),
        final_success=success[:, -100:].mean(),
        eval_success=np.nanmean([r['eval']['success'] for r in runs]),
        eval_steps=np.nanmean([r['eval']['avg_steps_success'] for r in runs]),
        eval_return=np.nanmean([r['eval']['avg_return'] for r in runs]),
        eval_optimal=np.nanmean([r['eval']['optimal'] for r in runs]),
        episodes_to_90=np.nanmean([v for v in
                                   (episodes_to_success(success[i])
                                    for i in range(len(runs))) if not np.isnan(v)]
                                  ) if any(not np.isnan(episodes_to_success(success[i]))
                                           for i in range(len(runs))) else np.nan,
    )


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def plot_curves(summary, episodes, fname, title_suffix='', window=50):
    # 纵向三行布局：每个子图在 Word 中可占满页宽，约为横排的 3 倍大
    fig, axes = plt.subplots(3, 1, figsize=(7, 11))
    x = np.arange(episodes)
    for name, agg in summary.items():
        spec = ALGO_SPECS[name]
        c, label = spec['color'], spec['title']
        rm = moving_average(agg['rewards_mean'], window)
        rs = moving_average(agg['rewards_std'], window)
        xt = np.arange(window - 1, episodes)
        axes[0].plot(xt, rm, color=c, label=label, lw=1.8)
        axes[0].fill_between(xt, rm - rs, rm + rs, color=c, alpha=0.12)

        sm = moving_average(agg['success_mean'], 100)
        ss = moving_average(agg['success_std'], 100)
        xt2 = np.arange(99, episodes)
        axes[1].plot(xt2, sm, color=c, label=label, lw=1.8)
        axes[1].fill_between(xt2, np.clip(sm - ss, 0, 1), np.clip(sm + ss, 0, 1),
                             color=c, alpha=0.12)

        pm = moving_average(agg['steps_mean'], window)
        ps = moving_average(agg['steps_std'], window)
        axes[2].plot(xt, pm, color=c, label=label, lw=1.8)
        axes[2].fill_between(xt, np.clip(pm - ps, 0, None),
                             pm + ps, color=c, alpha=0.12)

    axes[0].set_title(f'回合回报 (滑动平均 w={window}){title_suffix}')
    axes[0].set_xlabel('回合'); axes[0].set_ylabel('总回报')
    axes[0].axhline(94, color='k', ls='--', lw=1, alpha=0.5)
    axes[0].text(5, 94, '最优 94', fontsize=8, va='bottom')

    axes[1].set_title('成功率 (滑动平均 w=100)' + title_suffix)
    axes[1].set_xlabel('回合'); axes[1].set_ylabel('成功率')
    axes[1].set_ylim(-0.03, 1.03)

    axes[2].set_title(f'回合步数 (滑动平均 w={window}){title_suffix}')
    axes[2].set_xlabel('回合'); axes[2].set_ylabel('步数')
    axes[2].axhline(OPTIMAL_STEPS, color='k', ls='--', lw=1, alpha=0.5)
    axes[2].text(5, OPTIMAL_STEPS, '最优 7 步', fontsize=8, va='bottom')

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_final_bars(summary, fname):
    names = list(summary.keys())
    labels = [ALGO_SPECS[n]['title'] for n in names]
    colors = [ALGO_SPECS[n]['color'] for n in names]

    # 2x2 布局：每个子图在 Word 中约为横排的 2 倍大
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = np.asarray(axes).ravel()
    stats = [
        ('final_reward', '最后100回合平均回报', None),
        ('final_success', '最后100回合成功率', (0, 1.05)),
        ('eval_steps', '贪心评测平均步数', None),
        ('eval_optimal', '贪心评测最优路径比例', (0, 1.05)),
    ]
    for ax, (key, title, ylim) in zip(axes, stats):
        vals = [summary[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=8)
        ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_policy_grids(agents, env, fname, path):
    """Greedy policy arrows of each algorithm on the maze grid."""
    n = len(agents)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.1 * nrows))
    axes = np.array(axes).reshape(-1)
    arrow = {0: (0, 0.35), 1: (0.35, 0), 2: (0, -0.35), 3: (-0.35, 0)}

    for ax, (name, agent) in zip(axes, agents.items()):
        ax.set_xlim(0, env.size); ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(env.size) + 0.5)
        ax.set_yticks(np.arange(env.size) + 0.5)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) in env.traps:
                    fc = 'red'
                elif (i, j) == env.target_pos:
                    fc = 'yellow'
                else:
                    fc = 'white'
                ax.add_patch(plt.Rectangle((j, env.size - i - 1), 1, 1,
                                           edgecolor='gray', facecolor=fc))
        # optimal path overlay
        pr = [p[0] for p in path]; pc = [p[1] for p in path]
        ax.plot([c + 0.5 for c in pc], [env.size - r - 0.5 for r in pr],
                'g-', lw=2.5, alpha=0.55, zorder=1)

        if hasattr(agent, 'q_table'):
            visited = set(agent.q_table.keys())
        elif hasattr(agent, 'q_table1'):  # Double Q: union of both tables
            visited = set(agent.q_table1.keys()) | set(agent.q_table2.keys())
        elif hasattr(agent, 'h_table'):   # REINFORCE: preference table
            visited = set(agent.h_table.keys())
        else:
            visited = set()
        for (i, j) in visited:
            if (i, j) in env.traps or (i, j) == env.target_pos:
                continue
            va = env.get_valid_actions((i, j))
            a = agent.get_greedy_action((i, j), va)
            dx, dy = arrow[a]
            ax.arrow(j + 0.5, env.size - i - 0.5, dx, dy, head_width=0.13,
                     head_length=0.13, fc='black', ec='black', zorder=3)
        ax.set_title(ALGO_SPECS[name]['title'], fontsize=11)

    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('各算法学到的贪心策略（绿线 = BFS 最短路径）', y=1.0)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def experiment_main(episodes, seeds, outdir):
    print(f"\n=== 主实验: 全部 {len(MAIN_ORDER)} 种算法 | {episodes} 回合 x {seeds} 个种子 ===")
    env = SimpleMaze()
    path = optimal_path(env)
    v_star = value_iteration(env, gamma=0.95)
    print(f"最短安全路径: {path}  ({len(path) - 1} 步)")
    print("V*(起点) =", round(v_star[env.start_pos], 3))

    summary, final_agents, probe_curves = {}, {}, {}
    for name in MAIN_ORDER:
        t0 = time.time()
        runs = []
        for seed in range(seeds):
            res, agent = train_one(name, episodes, seed,
                                   probe_every=max(1, episodes // 200),
                                   probe_path=path)
            runs.append(res)
        summary[name] = summarize(runs)
        final_agents[name] = agent
        if runs[0]['probes'] is not None:
            probe_curves[name] = np.array([r['probes'] for r in runs])
        agg = summary[name]
        print(f"  {ALGO_SPECS[name]['title']:<28} final_return={agg['final_reward']:7.2f} "
              f"final_success={agg['final_success']:.2f} eval_steps={agg['eval_steps']:5.1f} "
              f"eval_optimal={agg['eval_optimal']:.2f} "
              f"episodes_to_90%={agg['episodes_to_90']:.0f}  ({time.time() - t0:.1f}s)")

    np.savez(os.path.join(outdir, 'main_results.npz'),
             **{f'{n}_{k}': v for n, agg in summary.items()
                for k, v in agg.items() if isinstance(v, np.ndarray)})

    plot_curves(summary, episodes, os.path.join(outdir, 'fig1_main_curves.png'))
    plot_final_bars(summary, os.path.join(outdir, 'fig2_main_final_metrics.png'))
    plot_policy_grids(final_agents, env, os.path.join(outdir, 'fig3_policy_grids.png'), path)
    return summary, final_agents, path, v_star, probe_curves


def experiment_overestimation(probe_curves, episodes, seeds, outdir, path, v_star):
    print(f"\n=== 过估计偏差实验: Q-Learning vs Double Q-Learning ===")
    curves = {n: probe_curves[n] for n in ['q_learning', 'double_q_learning']}
    probe_every = max(1, episodes // 200)
    xs = np.arange(0, episodes, probe_every)
    true_vals = [v_star[s] for s in path if s != SimpleMaze().target_pos]
    v_true_mean = float(np.mean(true_vals))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, mat in curves.items():
        m, s = mat.mean(0), mat.std(0)
        ax.plot(xs, m, color=ALGO_SPECS[name]['color'], lw=2,
                label=ALGO_SPECS[name]['title'])
        ax.fill_between(xs, m - s, m + s, color=ALGO_SPECS[name]['color'], alpha=0.15)
    ax.axhline(v_true_mean, color='k', ls='--', lw=1.5,
               label=f'精确 V* 路径均值 ({v_true_mean:.1f})')
    ax.set_xlabel('回合'); ax.set_ylabel('路径 states 的平均 max_a Q(s,a)')
    ax.set_title('价值估计收敛过程: Q-Learning vs Double Q-Learning (对照精确 V*)')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig4_overestimation.png'), dpi=150)
    plt.close(fig)
    print(f"  精确 V* 路径均值 = {v_true_mean:.2f}; "
          f"Q-Learning 末端 = {curves['q_learning'][:, -1].mean():.2f}, "
          f"Double Q 末端 = {curves['double_q_learning'][:, -1].mean():.2f}")

    # Per-state table: learned max-Q vs exact V* (10 fresh seeds)
    n_tab = min(10, seeds)
    trained = {n: [] for n in curves}
    for seed in range(n_tab):
        for n in trained:
            trained[n].append(train_one(n, episodes, seed)[1])
    with open(os.path.join(outdir, 'per_state_q_values.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['state', 'V_star', 'QLearning_maxQ_mean', 'QLearning_maxQ_std',
                    'DoubleQ_maxQ_mean', 'DoubleQ_maxQ_std'])
        for s in [st for st in path if st != SimpleMaze().target_pos]:
            va = SimpleMaze().get_valid_actions(s)
            q = np.array([max(a.q_table[s][x] for x in va) for a in trained['q_learning']])
            d = np.array([max(((a.q_table1[s] + a.q_table2[s]) / 2.0)[x] for x in va)
                              for a in trained['double_q_learning']])
            w.writerow([str(s), f'{v_star[s]:.2f}', f'{q.mean():.2f}', f'{q.std():.2f}',
                        f'{d.mean():.2f}', f'{d.std():.2f}'])
    print(f"  逐状态对照表已保存: per_state_q_values.csv ({n_tab} seeds)")


def experiment_alpha(episodes, seeds, outdir):
    print(f"\n=== 参数实验: 学习率 α (Q-Learning / SARSA / Expected SARSA) ===")
    alphas = [0.05, 0.1, 0.3, 0.6]
    # 纵向三行布局，放大每个子图
    fig, axes = plt.subplots(3, 1, figsize=(7, 11))
    for ax, name in zip(axes, ['q_learning', 'sarsa', 'expected_sarsa']):
        for a in alphas:
            runs = [train_one(name, episodes, s, overrides={'learning_rate': a})[0]
                    for s in range(seeds)]
            agg = summarize(runs)
            rm = moving_average(agg['rewards_mean'], 50)
            rs = moving_average(agg['rewards_std'], 50)
            xt = np.arange(49, episodes)
            ax.plot(xt, rm, lw=1.8, label=f'α={a}')
            ax.fill_between(xt, rm - rs, rm + rs, alpha=0.12)
        ax.set_title(f"{ALGO_SPECS[name]['title']}: 学习率的影响")
        ax.set_xlabel('回合'); ax.set_ylabel('平均回报 (MA50)')
        ax.axhline(94, color='k', ls='--', lw=1, alpha=0.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig5_alpha_sweep.png'), dpi=150)
    plt.close(fig)


def experiment_epsilon(episodes, seeds, outdir):
    print(f"\n=== 参数实验: 探索策略 (Q-Learning / SARSA) ===")
    schedules = [
        (('exp', 1.0, 0.95, 0.01), 'ε: 1.0→0.01 (decay 0.95)'),
        (('exp', 1.0, 0.98, 0.01), 'ε: 1.0→0.01 (decay 0.98)'),
        (('exp', 1.0, 0.995, 0.01), 'ε: 1.0→0.01 (decay 0.995)'),
        (('fixed', 0.1), 'ε 恒定 0.1'),
        (('glie',), 'ε = 1/k (GLIE)'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, name in zip(axes, ['q_learning', 'sarsa']):
        for sched, label in schedules:
            runs = [train_one(name, episodes, s, eps_schedule=sched)[0]
                    for s in range(seeds)]
            agg = summarize(runs)
            sm = moving_average(agg['success_mean'], 100)
            ss = moving_average(agg['success_std'], 100)
            xt = np.arange(99, episodes)
            ax.plot(xt, sm, lw=1.8, label=label)
            ax.fill_between(xt, np.clip(sm - ss, 0, 1), np.clip(sm + ss, 0, 1), alpha=0.12)
        ax.set_title(f"{ALGO_SPECS[name]['title']}: 探索 schedule 的影响")
        ax.set_xlabel('回合'); ax.set_ylabel('成功率 (MA100)')
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig6_epsilon_sweep.png'), dpi=150)
    plt.close(fig)


def experiment_lambda(episodes, seeds, outdir):
    print(f"\n=== 参数实验: Sarsa(λ) 的 λ ===")
    lambdas = [0.0, 0.3, 0.6, 0.9, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    results = {}
    for lam in lambdas:
        runs = [train_one('sarsa_lambda', episodes, s, overrides={'lam': lam})[0]
                for s in range(seeds)]
        results[lam] = summarize(runs)
        rm = moving_average(results[lam]['rewards_mean'], 50)
        rs = moving_average(results[lam]['rewards_std'], 50)
        xt = np.arange(49, episodes)
        axes[0].plot(xt, rm, lw=1.8, label=f'λ={lam}')
        axes[0].fill_between(xt, rm - rs, rm + rs, alpha=0.12)
        sm = moving_average(results[lam]['success_mean'], 100)
        xt2 = np.arange(99, episodes)
        axes[1].plot(xt2, sm, lw=1.8, label=f'λ={lam}')
    axes[0].set_title('Sarsa(λ): λ 对回报的影响 (λ=0 即 SARSA)')
    axes[0].set_xlabel('回合'); axes[0].set_ylabel('平均回报 (MA50)')
    axes[0].axhline(94, color='k', ls='--', lw=1, alpha=0.5)
    axes[1].set_title('Sarsa(λ): λ 对成功率的影响')
    axes[1].set_xlabel('回合'); axes[1].set_ylabel('成功率 (MA100)')
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig7_lambda_sweep.png'), dpi=150)
    plt.close(fig)
    for lam in lambdas:
        agg = results[lam]
        print(f"  λ={lam:<4} final_return={agg['final_reward']:7.2f} "
              f"episodes_to_90%={agg['episodes_to_90']:.0f}")


def experiment_reinforce(episodes, seeds, outdir):
    print(f"\n=== 消融实验: REINFORCE 有无 baseline ===")
    variants = [('reinforce', dict(use_baseline=True), 'REINFORCE + baseline'),
                ('reinforce', dict(use_baseline=False), 'REINFORCE (无 baseline)')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for name, over, label in variants:
        runs = [train_one(name, episodes, s, overrides=over)[0] for s in range(seeds)]
        agg = summarize(runs)
        rm = moving_average(agg['rewards_mean'], 50)
        rs = moving_average(agg['rewards_std'], 50)
        xt = np.arange(49, episodes)
        axes[0].plot(xt, rm, lw=1.8, label=label)
        axes[0].fill_between(xt, rm - rs, rm + rs, alpha=0.15)
        sm = moving_average(agg['success_mean'], 100)
        xt2 = np.arange(99, episodes)
        axes[1].plot(xt2, sm, lw=1.8, label=label)
        print(f"  {label:<26} final_return={agg['final_reward']:7.2f} "
              f"episodes_to_90%={agg['episodes_to_90']:.0f}")
    axes[0].set_title('REINFORCE: baseline 对回报的影响')
    axes[0].set_xlabel('回合'); axes[0].set_ylabel('平均回报 (MA50)')
    axes[0].axhline(94, color='k', ls='--', lw=1, alpha=0.5)
    axes[1].set_title('REINFORCE: baseline 对成功率的影响')
    axes[1].set_xlabel('回合'); axes[1].set_ylabel('成功率 (MA100)')
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig8_reinforce_baseline.png'), dpi=150)
    plt.close(fig)


def write_summary_csv(summary, fname):
    keys = ['final_reward', 'final_reward_std', 'final_success', 'eval_success',
            'eval_steps', 'eval_return', 'eval_optimal', 'episodes_to_90']
    with open(fname, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'category'] + keys)
        for name, agg in summary.items():
            row = [ALGO_SPECS[name]['title'], ALGO_SPECS[name]['category']]
            row += [f"{agg[k]:.2f}" if not np.isnan(agg[k]) else 'nan' for k in keys]
            w.writerow(row)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Tabular RL algorithm comparison on SimpleMaze')
    parser.add_argument('--experiment', default='all',
                        choices=['main', 'overestimation', 'alpha', 'epsilon',
                                 'lambda', 'reinforce', 'all'])
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--outdir', default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    summary = path = v_star = probe_curves = None
    if args.experiment in ('main', 'all'):
        summary, _, path, v_star, probe_curves = experiment_main(
            args.episodes, args.seeds, args.outdir)
        write_summary_csv(summary, os.path.join(args.outdir, 'summary.csv'))

    if args.experiment in ('overestimation', 'all'):
        if path is None:
            env = SimpleMaze()
            path = optimal_path(env)
            v_star = value_iteration(env, gamma=0.95)
            probe_curves = {}
            for name in ['q_learning', 'double_q_learning']:
                runs = [train_one(name, args.episodes, s,
                                  probe_every=max(1, args.episodes // 200),
                                  probe_path=path)[0] for s in range(args.seeds)]
                probe_curves[name] = np.array([r['probes'] for r in runs])
        experiment_overestimation(probe_curves, args.episodes, args.seeds,
                                  args.outdir, path, v_star)

    if args.experiment in ('alpha', 'all'):
        experiment_alpha(args.episodes, args.seeds, args.outdir)

    if args.experiment in ('epsilon', 'all'):
        experiment_epsilon(args.episodes, args.seeds, args.outdir)

    if args.experiment in ('lambda', 'all'):
        experiment_lambda(args.episodes, args.seeds, args.outdir)

    if args.experiment in ('reinforce', 'all'):
        experiment_reinforce(args.episodes, args.seeds, args.outdir)

    print(f"\n全部完成, 用时 {time.time() - t_start:.1f}s. 图表保存在: {args.outdir}")


if __name__ == '__main__':
    main()
