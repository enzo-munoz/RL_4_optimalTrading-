import numpy as np

# -------------------------------------------------------------------
# Model Architecture Constants
# -------------------------------------------------------------------

# Per-case Actor/Critic hidden layer configurations (list of hidden dims)
# Case 1 (MC θ only):    5 layers, 16 nodes
# Case 2 (MC θ, κ):      5 layers, 20 nodes
# Case 3 (MC θ, κ, σ):   6 layers, 20 nodes
CASE_ARCHITECTURES = {
    1: [16, 16, 16, 16, 16],
    2: [20, 20, 20, 20, 20],
    3: [20, 20, 20, 20, 20, 20],
}

# Default architecture (Case 1)
HIDDEN_LAYERS     = CASE_ARCHITECTURES[1]
HIDDEN_DIM        = HIDDEN_LAYERS[0]
NUM_HIDDEN_LAYERS = len(HIDDEN_LAYERS)

# GRU recurrent layers / hidden dim (Table 2)
GRU_HIDDEN_LAYERS = 5
GRU_HIDDEN_DIM    = 20

# Lookback windows (Table 2): W=10 for both prob-DDPG and reg-DDPG
LOOKBACK_WINDOW_PROB = 10
LOOKBACK_WINDOW_REG  = 10
LOOKBACK_WINDOW      = LOOKBACK_WINDOW_PROB   # backward-compat alias

GRU_OUTPUT_SIZE = 1
MAX_ACTION      = 10.0


class SimulationConfig:
    MAX_ACTION = 10.0

    # Inventory bounds
    I_max = MAX_ACTION
    I_min = -MAX_ACTION

    # Trading / simulation
    transaction_cost = 0.05   # λ
    dt               = 0.2    # Δt
    n_steps          = 2000   # steps per episode
    mu_inv           = 1.0    # invariant mean
    sigma_inv        = 0.1    # invariant std dev — used for S_0 ~ N(µ_inv, 3·σ_inv)

    # Per-mode lookback windows
    lookback_window_prob = LOOKBACK_WINDOW_PROB   # W = 10 (prob-DDPG)
    lookback_window_reg  = LOOKBACK_WINDOW_REG    # W = 10 (reg-DDPG)
    lookback_window      = LOOKBACK_WINDOW_PROB   # default (overridden by mode)

    # Regime values
    theta_values = [0.9, 1.0, 1.1]
    kappa_values = [3.0, 7.0]
    sigma_values = [0.1, 0.3]

    # Transition rate matrices
    A_theta = np.array([
        [-0.1,  0.05,  0.05],
        [ 0.05, -0.1,  0.05],
        [ 0.05,  0.05, -0.1],
    ])
    A_kappa = np.array([[-0.1, 0.1], [0.1, -0.1]])
    A_sigma = np.array([[-0.1, 0.1], [0.1, -0.1]])

    # Training hyperparameters (Table 1)
    n_episodes = 10000
    batch_size = 51
    lr         = 0.01
    tau        = 0.05   # soft-update rate
    gamma      = 0.99
    a          = 100     # ε decays linearly to 0.01 over the first `a` iterations

    # Test
    n_test_episodes = 500

    # Per-case Actor/Critic architecture (used by DDPGAgent)
    case_architectures = CASE_ARCHITECTURES
