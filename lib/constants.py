import numpy as np
# -------------------------------------------------------------------
# Model Architecture Constants
# -------------------------------------------------------------------

# Dimensions des couches cachées
HIDDEN_DIM = 20

# Configuration des couches cachées (nombre de neurones par couche)
# Architecture: Input -> 20 -> 20 -> 20 -> Output
HIDDEN_LAYERS = [20, 20, 20] 

# Nombre total de couches cachées (dérivé de HIDDEN_LAYERS si possible, ou défini explicitement)
NUM_HIDDEN_LAYERS = len(HIDDEN_LAYERS)
GRU_HIDDEN_LAYERS = 2
GRU_HIDDEN_DIM = 1

LOOKBACK_WINDOW = 10

GRU_OUTPUT_SIZE = 1
MAX_ACTION = 10.0

class SimulationConfig:
    MAX_ACTION = 10.0
    # Inventaire
    I_max = MAX_ACTION
    I_min = -MAX_ACTION
    
    # Trading
    transaction_cost = 0.05  # lambda
    dt = 0.2  # Delta t
    n_steps = 2000  # pas par épisode
    mu_inv = 1.0  # moyenne invariante
    lookback_window = LOOKBACK_WINDOW
    
    # Régimes theta
    theta_values = [0.9, 1.0, 1.1]
    kappa_values = [3.0, 7.0]
    sigma_values = [0.1, 0.3]
    
    # Matrices de taux de transition
    A_theta = np.array([
        [-0.1, 0.05, 0.05],
        [0.05, -0.1, 0.05],
        [0.05, 0.05, -0.1]
    ])
    A_kappa = np.array([
        [-0.1, 0.1],
        [0.1, -0.1]
    ])
    A_sigma = np.array([
        [-0.1, 0.1],
        [0.1, -0.1]
    ])
    
    # Entraînement
    n_episodes = 10000
    batch_size = 512
    lr = 0.001
    tau = 0.001  # soft update
    gamma = 0.99  # facteur d'actualisation

    # Test
    n_test_episodes = 1
