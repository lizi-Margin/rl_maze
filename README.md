# 强化学习迷宫导航作业

这是一个强化学习实践作业，你需要实现Q-Learning和SARSA两种算法来训练智能体在迷宫环境中导航。

> **扩展工作（本仓库的创新点）**：在完成原始作业（Q-Learning / SARSA）的基础上，本仓库进一步实现了 **8 种经典无模型（model-free）控制算法**，分为"基于价值（Q）"与"基于策略（policy）"两大类，在同一环境、同一评测协议（相同随机种子、统一 ε 调度、贪心策略独立评测）下系统对比，并完成了 α / ε / λ / baseline 共 4 组参数敏感性实验。全部实验一键复现，详细分析与图表见 [REPORT.md](REPORT.md)。

## 📁 项目结构

```
├── environment.py            # 迷宫环境和可视化（无需修改）
├── q_learning.py             # Q-Learning算法（模板TODO已补全）
├── sarsa.py                  # SARSA算法（模板TODO已补全）
├── expected_sarsa.py         # 扩展：Expected SARSA（value 类）
├── double_q_learning.py      # 扩展：Double Q-Learning（value 类）
├── sarsa_lambda.py           # 扩展：Sarsa(λ) 资格迹（value 类）
├── mc_control.py             # 扩展：同轨首次访问 MC 控制（policy 类）
├── offpolicy_mc_control.py   # 扩展：离轨 MC 控制 + 加权重要性采样（policy 类）
├── reinforce.py              # 扩展：REINFORCE + 蒙特卡洛 baseline（policy 类）
├── run_experiments.py        # 扩展：统一评测框架（多随机种子 + 参数实验）
├── train.py                  # 原始交互式训练/测试接口（无需修改）
├── REPORT.md                 # 扩展：8 算法对比分析报告
├── results/                  # 扩展：全部实验图表与数据（fig1~fig8、summary.csv 等）
└── README.md                 # 本文件
```

## 🎮 环境说明

迷宫是一个5×5的网格，包含：
- **白色方块**：空地（安全移动）
- **红色方块**：陷阱（-50奖励，回合结束）
- **黄色方块**：目标（+100奖励，回合结束）
- **黑色圆圈**：智能体（从左上角开始）

## 📝 作业要求

### 1. Q-Learning算法实现（`q_learning.py`）

Q-Learning是一种off-policy算法，学习最优Q函数：

**更新规则：**
```
Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
```

**需要完成的任务：**
1. 在`QLearningAgent`类中实现`learn()`方法
2. （选做）尝试不同的探索策略

### 2. SARSA算法实现（`sarsa.py`）

SARSA是一种on-policy算法，学习执行策略的Q函数：

**更新规则：**
```
Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
```

**需要完成的任务：**
1. 在`SARSAAgent`类中实现`learn()`方法
2. （选做）尝试不同的探索策略

## 🚀 环境配置

### 安装依赖包

```bash
pip install numpy matplotlib
```

### 运行原始交互作业

```bash
python train.py
```

这会显示一个菜单，包含以下选项：
- 训练和测试Q-Learning智能体
- 训练和测试SARSA智能体
- 对比两种算法
- 快速训练（无可视化）

### 运行扩展对比实验（创新点）

```bash
# 一键复现全部结果：主对比 + 4 组参数实验（约 1 分钟）
python run_experiments.py --experiment all --episodes 2000 --seeds 20

# 或单独跑某一个实验
python run_experiments.py --experiment main|overestimation|alpha|epsilon|lambda|reinforce
```

图表与数据保存在 `results/` 目录，算法设计与结果分析见 [REPORT.md](REPORT.md)。

## 📋 实现步骤

### 第一步：实现Q-Learning

1. 打开`q_learning.py`文件
2. 找到`QLearningAgent`类中的`learn()`方法
3. 实现Q-learning更新规则
4. 使用`train.py`测试你的实现

### 第二步：实现SARSA

1. 打开`sarsa.py`文件
2. 找到`SARSAAgent`类中的`learn()`方法
3. 实现SARSA更新规则
4. 使用`train.py`测试你的实现

### 第三步：对比算法

使用`train.py`中的对比功能来观察Q-Learning和SARSA的差异：
- Q-Learning学习最优策略（可能更冒险）
- SARSA学习更安全的策略（考虑探索）

## 🔍 关键概念

### 状态表示
- 状态用元组`(row, column)`表示智能体位置

### 动作空间
- 4个离散动作：0=上, 1=右, 2=下, 3=左

### 奖励设计
- 到达目标：+100（回合结束）
- 踩到陷阱：-50（回合结束）
- 每步移动：-1（鼓励走最短路径）

### 探索与利用
- 使用ε-贪婪策略(可自行尝试替换)
- ε随时间从1.0衰减到0.01
- 测试时ε设为0（纯贪心策略）


## 📊 提交要求

提交作业时，请确保：

1. `q_learning.py`和`sarsa.py`中的`learn()`方法已完整实现
2. 代码包含必要的注释
3. 提交一份简短的报告，包括：
   - 学习曲线（奖励随回合的变化）
   - 成功率统计
   - 两种算法行为的差异讨论
   - 实现过程中的心得体会

扩展部分的完整报告（含 8 算法对比表、8 张图表、参数敏感性分析与复现命令）见 [REPORT.md](REPORT.md)。

祝你完成顺利！如有问题，请及时联系助教( 202421081056@mail.bnu.edu.cn )。
