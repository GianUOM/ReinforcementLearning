# Reinforcement Learning for Blackjack

A comparative study of **Q-learning**, **SARSA**, and **Monte Carlo** to train agents to play **Blackjack** through self-play. Agents learn when to **HIT** or **STAND** using different learning methods and exploration strategies.

---

## Algorithms Implemented

### 1. Q-Learning (Off-policy TD Control)
- Learns by estimating the **maximum future reward**.
- Supports multiple ε-greedy exploration strategies:
  - `ε = 0.1` (constant)
  - `ε = 1/k` (decay)
  - `ε = e^(-k/1000)`, `ε = e^(-k/10000)` (exponential decay)

### 2. SARSA (On-policy TD Control)
- Learns using the **actual next action** taken.
- Trained with the same four ε strategies as Q-learning.

### 3. Monte Carlo (First-Visit)
- Learns by **completing entire episodes** before updating.
- Two approaches:
  - **Exploring Starts (ES)** – Forces exploration from all states
  - **Non-Exploring Starts (NES)** – Uses ε-greedy with decay

---

## Objectives

- Learn optimal HIT/STAND strategy for Blackjack.
- Compare different RL methods and exploration strategies.
- Evaluate using:
  - Win/Loss/Draw statistics
  - State-action space coverage
  - Strategy tables
  - Dealer advantage

---

## Visualizations

Each method includes:

- **Win/Loss/Draw graphs**
- **State-action pair bar plots**
- **Strategy tables** (with/without Ace as 11)
- **Dealer Advantage** comparison across methods

---

## 📁 Project Structure

```bash
Blackjack-RL
├── Qlearningnew.py          # Q-learning implementation
├── Sarsanew.py              # SARSA implementation
├── main.py                  # Monte Carlo + evaluation/visualizations
└── README.md                # You're here!
