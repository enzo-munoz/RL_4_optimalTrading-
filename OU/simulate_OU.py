import numpy as np
import os
import sys
from typing import List, Tuple

# Add parent directory to path to import SimulationConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constants import SimulationConfig

class MarkovChain:
    """Chaîne de Markov à temps continu"""
    def __init__(self, states: List[float], rate_matrix: np.ndarray, dt: float):
        self.states = np.array(states)
        self.rate_matrix = rate_matrix
        self.dt = dt
        self.n_states = len(states)
        self.current_idx = np.random.randint(self.n_states)
        
        # Matrice de transition pour dt
        self.trans_matrix = np.eye(self.n_states) + rate_matrix * dt
        
    def step(self) -> float:
        probs = self.trans_matrix[self.current_idx]
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()
        self.current_idx = np.random.choice(self.n_states, p=probs)
        return self.states[self.current_idx]
    
    def reset(self):
        self.current_idx = np.random.randint(self.n_states)
        return self.states[self.current_idx]

class OUProcess:
    """Processus Ornstein-Uhlenbeck avec régimes Markoviens"""
    def __init__(self, config: SimulationConfig, case: int = 1):
        self.config = config
        self.case = case
        self.dt = config.dt
        
        # Initialiser les chaînes de Markov selon le cas
        self.theta_mc = MarkovChain(config.theta_values, config.A_theta, self.dt)
        
        if case >= 2:
            self.kappa_mc = MarkovChain(config.kappa_values, config.A_kappa, self.dt)
        else:
            self.kappa_mc = None
            self.kappa_fixed = 5.0
            
        if case >= 3:
            self.sigma_mc = MarkovChain(config.sigma_values, config.A_sigma, self.dt)
        else:
            self.sigma_mc = None
            self.sigma_fixed = 0.2
            
        self.S = config.mu_inv
        
    def step(self) -> Tuple[float, float, float, float]:
        theta = self.theta_mc.step()
        kappa = self.kappa_mc.step() if self.kappa_mc else self.kappa_fixed
        sigma = self.sigma_mc.step() if self.sigma_mc else self.sigma_fixed
        
        dW = np.random.normal(0, np.sqrt(self.dt))
        dS = kappa * (theta - self.S) * self.dt + sigma * dW
        self.S += dS
        
        return self.S, theta, kappa, sigma

    def reset(self) -> float:
        self.theta_mc.reset()
        if self.kappa_mc:
            self.kappa_mc.reset()
        if self.sigma_mc:
            self.sigma_mc.reset()
        self.S = self.config.mu_inv
        return self.S

def simulate_and_save(config: SimulationConfig, case: int):
    """
    Simule et enregistre les données pour un cas spécifique.
    """
    # Définir le répertoire de sortie en fonction du cas
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'replay_buffer', 'data'))
    
    if case == 1:
        output_dir = os.path.join(base_dir, 'theta_MK')
    elif case == 2:
        output_dir = os.path.join(base_dir, 'theta_kappa_MK')
    elif case == 3:
        output_dir = os.path.join(base_dir, 'theta_kappa_sigma_MK')
    else:
        raise ValueError(f"Cas inconnu: {case}")
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Lancement de la simulation pour le Cas {case}. Sortie: {output_dir}")
    
    env = OUProcess(config, case=case)
    
    # Utiliser n_test_episodes pour la génération de données (ou n_episodes si nécessaire)
    n_episodes = config.n_test_episodes
    n_steps = config.n_steps
    
    for episode in range(n_episodes):
        env.reset()
        data = []
        for step in range(n_steps):
            S, theta, kappa, sigma = env.step()
            # Enregistrer temps, prix, et les paramètres du régime
            data.append([step * config.dt, S, theta, kappa, sigma])
            
        # Sauvegarder en CSV
        filename = os.path.join(output_dir, f"episode_{episode}.csv")
        header = "t,S,theta,kappa,sigma"
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        
        if (episode + 1) % 50 == 0:
            print(f"  Cas {case}: Épisode {episode + 1}/{n_episodes} sauvegardé.")

if __name__ == "__main__":
    config = SimulationConfig()
    
    # Exécuter pour les 3 cas
    for case in [1, 2, 3]:
        simulate_and_save(config, case)
