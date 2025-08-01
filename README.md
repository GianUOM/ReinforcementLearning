# Reinforcement Learning in Blackjack

This project compares three classic reinforcement learning algorithms — Monte Carlo, SARSA, and Q-Learning — to train agents to play Blackjack through self-play. It was implemented from scratch as part of an academic project at the University of Malta.

---

## Project Summary

A custom Blackjack environment was created, and three agents were trained to learn HIT/STAND strategies using different learning methods and exploration strategies. Performance was analyzed across 100,000 episodes per method, with strategy tables, win/loss curves, and state-action exploration visualized.

---

## Algorithms Implemented

### Q-Learning (Off-policy)
- Learns by estimating the maximum expected future reward for the next state, regardless of the agent's actual action.
- Filename: `Qlearningnew.py`

### SARSA (On-policy)
- Updates Q-values using the actual action taken in the next state, making it more conservative and stable early in training.
- Filename: `Sarsanew.py`

### Monte Carlo (First-Visit)
- Updates are made after full episodes based on first visits to state-action pairs.
- Two approaches included:
  - Exploring Starts (ES)
  - Non-Exploring Starts (NES) with epsilon decay strategies
- Filename: `monteCarlo.py`

---

## Key Features

- Custom-built Blackjack game environment (no external RL or Gym libraries)
- Multiple epsilon decay strategies for exploration:
  - ε = 0.1 (constant)
  - ε = 1/k (inverse)
  - ε = e^(-k/1000), ε = e^(-k/10000) (exponential)
- Tracks win, loss, and draw counts
- Tracks state-action pair visit counts
- Generates strategy tables for hard and soft hands
- Compares dealer advantage across all methods

---

## Repository Contents

```bash
ReinforcementLearning/
├── Qlearningnew.py                           # Q-learning agent
├── Sarsanew.py                               # SARSA agent
├── monteCarlo.py                             # Monte Carlo agent and analysis
├── ReinforcementLearningGianlucaAquilina...  # Final academic report (PDF)
├── README.md                                 # Project documentation
