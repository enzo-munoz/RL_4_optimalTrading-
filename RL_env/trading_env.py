import numpy as np
from collections import deque
from typing import Dict, Tuple
import sys
import os

# Add project root to path if not already there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.constants import SimulationConfig
from OU.simulate_OU import OUProcess

class TradingEnvironment:
    def __init__(self, config: SimulationConfig, case: int = 3):
        self.config = config
        self.ou_process = OUProcess(config, case)
        self.lookback = config.lookback_window
        
        self.S_history = deque(maxlen=self.lookback)
        self.I = 0  # Inventaire
        self.t = 0
        
    def reset(self) -> Dict:
        self.ou_process.reset()
        self.I = np.random.uniform(self.config.I_min, self.config.I_max)
        self.t = 0
        
        # Remplir l'historique
        self.S_history.clear()
        S = self.ou_process.S
        for _ in range(self.lookback):
            S, _, _, _ = self.ou_process.step()
            self.S_history.append(S)
            
        return self._get_state()
    
    def _get_state(self) -> Dict:
        return {
            'S': self.ou_process.S,
            'I': self.I,
            'S_history': np.array(self.S_history),
            't': self.t
        }
    
    def step(self, action: float) -> Tuple[Dict, float, bool]:
        """
        Action: changement d'inventaire delta_I dans [-1, 1] (normalisé)
        """
        # Dénormaliser l'action
        delta_I = action * (self.config.I_max - self.config.I_min) / 2
        
        # Appliquer les contraintes d'inventaire
        new_I = np.clip(self.I + delta_I, self.config.I_min, self.config.I_max)
        actual_delta = new_I - self.I
        
        # Avancer le processus
        S_old = self.ou_process.S
        S_new, theta, kappa, sigma = self.ou_process.step()
        self.S_history.append(S_new)
        
        # Calculer la récompense
        # r_t = I_t * (S_{t+1} - S_t) - lambda * |delta_I|
        pnl = self.I * (S_new - S_old)
        cost = self.config.transaction_cost * abs(actual_delta)
        reward = pnl - cost
        
        # Mettre à jour l'état
        self.I = new_I
        self.t += 1
        
        done = self.t >= self.config.n_steps
        
        return self._get_state(), reward, done
