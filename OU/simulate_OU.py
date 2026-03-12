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

def simulate_and_save(config: SimulationConfig, case: int, n_episodes: int = None, start_episode: int = 0):
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
    
    # Utiliser n_episodes si fourni, sinon config.n_test_episodes
    if n_episodes is None:
        n_episodes = config.n_test_episodes
    
    n_steps = config.n_steps
    
    for i in range(n_episodes):
        episode_idx = start_episode + i
        env.reset()
        data = []
        for step in range(n_steps):
            S, theta, kappa, sigma = env.step()
            # Generate random inventory between I_min and I_max
            I = np.random.uniform(config.I_min, config.I_max)
            # Enregistrer temps, prix, et les paramètres du régime
            data.append([step * config.dt, S, theta, kappa, sigma, I])
            
        # Sauvegarder en CSV
        filename = os.path.join(output_dir, f"episode_{episode_idx}.csv")
        header = "t,S,theta,kappa,sigma,I"
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        
        if (i + 1) % 50 == 0 or (i + 1) == n_episodes:
            print(f"  Cas {case}: Épisode {episode_idx} sauvegardé ({i + 1}/{n_episodes}).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate OU simulation episodes")
    parser.add_argument("--case",          type=int, default=1, choices=[1, 2, 3],
                        help="1=θ only  2=θ+κ  3=θ+κ+σ")
    parser.add_argument("--n_episodes",    type=int, default=500)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--test",          action="store_true",
                        help="Write to a separate _test subdirectory (held-out evaluation set)")
    args = parser.parse_args()

    config   = SimulationConfig()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'replay_buffer', 'data'))
    _dirs    = {1: 'theta_MK', 2: 'theta_kappa_MK', 3: 'theta_kappa_sigma_MK'}
    out_dir  = os.path.join(base_dir, _dirs[args.case] + ('_test' if args.test else ''))
    os.makedirs(out_dir, exist_ok=True)

    env = OUProcess(config, case=args.case)
    print(f"Generating {args.n_episodes} episodes -> {out_dir}")
    for i in range(args.n_episodes):
        ep_idx = args.start_episode + i
        env.reset()
        data = []
        for step in range(config.n_steps):
            S, theta, kappa, sigma = env.step()
            I = np.random.uniform(config.I_min, config.I_max)
            data.append([step * config.dt, S, theta, kappa, sigma, I])
        np.savetxt(
            os.path.join(out_dir, f"episode_{ep_idx}.csv"),
            data, delimiter=",", header="t,S,theta,kappa,sigma,I", comments="",
        )
        if (i + 1) % 50 == 0 or (i + 1) == args.n_episodes:
            print(f"  episode_{ep_idx}.csv  ({i+1}/{args.n_episodes})")
