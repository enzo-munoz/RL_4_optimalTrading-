# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Reinforcement Learning system for optimal inventory trading. It implements DDPG (Deep Deterministic Policy Gradient) agents augmented with pre-trained GRU encoders to trade a signal governed by a regime-switching Ornstein-Uhlenbeck process with hidden Markov parameters.

## Commands

### Setup
```bash
# Install dependencies using uv (project uses uv.lock)
uv sync

# Always invoke Python via the venv (system Python lacks the dependencies)
# Windows: .venv/Scripts/python
# Linux/Mac: .venv/bin/python
```

### Data Generation (must run before training)
```bash
# Generate OU simulation episodes for Case 1 (theta Markov only)
python OU/simulate_OU.py

# The script saves CSVs to replay_buffer/data/theta_MK/ (case 1),
# theta_kappa_MK/ (case 2), or theta_kappa_sigma_MK/ (case 3)
```

### GRU Pre-training (Step 1)
```bash
# Train regression GRU (predicts S_{t+1})
python models/train_GRU.py --type reg --data_dir replay_buffer/data/theta_MK --epochs 50

# Train probabilistic GRU (predicts theta regime probabilities)
python models/train_GRU.py --type prob --data_dir replay_buffer/data/theta_MK --epochs 50

# Outputs saved to models/checkpoints/best_gru_model_{type}.pth
```

### DDPG Training (Step 2)
```bash
# Train DDPG agent — no replay buffer, batches generated on-the-fly
python main.py --mode reg --case 1 --N 1000
python main.py --mode prob --case 1 --N 1000

# Cases: 1=θ only, 2=θ+κ, 3=θ+κ+σ
python main.py --mode reg --case 3 --N 1000
```

### Evaluation
```bash
# Evaluate trained GRU model
python models/eval_GRU.py

# Run inference with trained DDPG agent
python RL_env/trading_env.py
```

## Architecture

### Two-Step Pipeline

**Step 1 — GRU Pre-training** (`models/train_GRU.py`, `models/gru.py`):
- A GRU network is trained *offline* on the pre-generated OU episodes (all available CSVs, no cap).
- Optimizer: **W-ADAM** (`lib/win_adam.py`) — windowed Adam with window size = W.
- Two modes: `reg` (W=10, predicts S_{t+1} via MSE) and `prob` (W=10, predicts θ regime via CrossEntropy).
- Architecture: 5 GRU layers × 20 hidden units; 5-layer MLP head × 64 hidden units.
- Saved to `models/checkpoints/best_gru_model_{type}.pth`.

**Step 2 — DDPG Training** (`main.py`):
- **No replay buffer** — fresh batches generated on-the-fly each iteration (paper Section 3.2): S₀ ~ N(µ_inv, 3·σ_inv), I ~ U[I_min, I_max], then W+1 OU steps.
- The GRU is loaded and **frozen** (no gradient updates).
- State G = [S_norm, I_norm, GRU(S_history)], dim = `2 + gru_out_dim`.
- Actor outputs target inventory I ∈ [-10, 10]; normalised to [-1,1] for the critic.
- Reward: `r = I_{t+1}*(S_{t+1} - S_t) - λ|ΔI|` with λ=0.05 (inventory **after** action, Eq. 4).
- ε decays linearly from 1.0 → 0.01 over the first `a=100` training iterations.
- Actor/Critic architecture varies by case (see Key Hyperparameters below).
- Soft target network updates (τ=0.001) for both Actor and Critic.

### Module Map

| Module | Purpose |
|---|---|
| `lib/constants.py` | `SimulationConfig` dataclass with all hyperparameters; network architecture constants |
| `OU/simulate_OU.py` | `OUProcess` + `MarkovChain` — generates episode CSVs |
| `replay_buffer/data/` | Pre-generated episode CSVs (`t, S, theta, kappa, sigma, I`) |
| `models/gru.py` | `GRUNet` — modular GRU with `reg`, `prob`, or `hid` head |
| `models/train_GRU.py` | Offline GRU pre-training script |
| `models/actor.py` | DDPG Actor network |
| `models/critic.py` | DDPG Critic network |
| `main.py` | `DDPGAgent` + `ReplayDataLoader` + training loop |
| `RL_env/trading_env.py` | Inference-only environment — replays pre-generated episodes |

### Market Model (3 Cases)

The `OUProcess` supports three complexity levels controlled by the `case` parameter:
- **Case 1** (`theta_MK`): θ follows a Markov chain; κ and σ are fixed.
- **Case 2** (`theta_kappa_MK`): θ and κ both follow independent Markov chains.
- **Case 3** (`theta_kappa_sigma_MK`): θ, κ, and σ all follow independent Markov chains.

### Key Hyperparameters (from `SimulationConfig` / `lib/constants.py`)

| Parameter | Value | Note |
|---|---|---|
| `batch_size` | 51 | per training iteration |
| `lr` | 0.01 | Adam for Actor/Critic |
| `gamma` | 0.99 | discount factor |
| `tau` | 0.001 | soft-update rate |
| `a` | 100 | ε fully decays after `a` iterations |
| `n_test_episodes` (M) | 500 | held-out evaluation episodes |
| `dt` | 0.2 | Euler-Maruyama step |
| `n_steps` | 2000 | steps per episode |
| `GRU_HIDDEN_LAYERS` | 5 | GRU recurrent layers |
| `GRU_HIDDEN_DIM` | 20 | GRU hidden units |
| `GRU_FC_HIDDEN` | 64 | MLP head hidden units (5 layers) |
| `LOOKBACK_WINDOW_PROB` | 10 | W for prob-DDPG |
| `LOOKBACK_WINDOW_REG` | 10 | W for reg-DDPG |

**Per-case Actor/Critic architecture** (`CASE_ARCHITECTURES` in `lib/constants.py`):
- Case 1 (θ only): 5 layers × 16 hidden nodes
- Case 2 (θ, κ): 5 layers × 20 hidden nodes
- Case 3 (θ, κ, σ): 6 layers × 20 hidden nodes

### Data Flow

```
simulate_OU.py → replay_buffer/data/{case}/episode_*.csv
                         ↓
          train_GRU.py (offline, frozen after)
                         ↓
          models/checkpoints/best_gru_model_{type}.pth
                         ↓
          main.py (DDPGAgent loads GRU + trains Actor/Critic)
                         ↓
          models/checkpoints/ddpg_{mode}_best.pth
                         ↓
          RL_env/trading_env.py (inference on held-out episodes)
```
