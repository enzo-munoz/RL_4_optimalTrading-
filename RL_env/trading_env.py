"""
Trading Environment — Inference only.
Loads pre-generated OU episodes, runs the actor (+ frozen GRU) step by step,
and records cumulative rewards.
No critic needed at inference time.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.constants import SimulationConfig, GRU_HIDDEN_DIM, GRU_HIDDEN_LAYERS
from models.gru import GRUNet
from models.actor import Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TradingEnvironment:
    """
    Replays a pre-generated OU episode and lets the actor trade.
    State  G = [S_norm, I_norm, GRU(S_history)]
    Action a = target inventory I in [I_min, I_max] (actor output)
    Reward r = I*(S_{t+1} - S_t) - lambda*|delta_I|
    """

    def __init__(self, config: SimulationConfig, mode: str, model_path: str):
        self.config = config
        self.W      = config.lookback_window

        # ── Load frozen GRU ──
        gru_out_dim = 1 if mode == "reg" else len(config.theta_values)
        self.gru = GRUNet(1, GRU_HIDDEN_DIM, GRU_HIDDEN_LAYERS,
                          output_size=gru_out_dim, head_type=mode).to(DEVICE)

        # ── Load Actor (no critic needed at inference) ──
        state_dim  = 2 + gru_out_dim
        self.actor = Actor(state_dim, 1).to(DEVICE)

        ckpt = torch.load(model_path, map_location=DEVICE)
        self.gru.load_state_dict(ckpt['gru'])
        self.actor.load_state_dict(ckpt['actor'])
        self.gru.eval(); self.actor.eval()

    # ── State builder ────────────────────────────────────────────────────────

    def _build_state(self, S: float, I: float, history: np.ndarray) -> torch.Tensor:
        """Encode (S, I, S_history) into state tensor G of shape (1, state_dim)."""
        S_norm = (S - self.config.mu_inv) / 0.5
        I_norm = 2 * (I - self.config.I_min) / (self.config.I_max - self.config.I_min) - 1

        hist_t = torch.FloatTensor(history).unsqueeze(0).unsqueeze(-1).to(DEVICE)  # (1, W, 1)
        with torch.no_grad():
            phi = self.gru(hist_t).squeeze(0)  # (gru_out_dim,)

        return torch.cat([
            torch.tensor([S_norm, I_norm], dtype=torch.float32, device=DEVICE), phi
        ]).unsqueeze(0)  # (1, state_dim)

    # ── Run one episode ──────────────────────────────────────────────────────

    def run_episode(self, S_series: np.ndarray) -> dict:
        """
        Run the actor greedily on one OU price series.
        S_series : 1-D array of length T.
        Returns  : cumulative_reward, full cumulative curve, rewards, inventories.
        """
        T       = len(S_series)
        history = deque(S_series[:self.W], maxlen=self.W)  # seed history with first W prices
        I       = np.random.uniform(self.config.I_min, self.config.I_max)

        rewards, inventories = [], []
        price_diffs = []

        for t in range(self.W, T - 1):
            S_curr = S_series[t]
            S_next = S_series[t + 1]
            diff_S = S_next - S_curr

            # Actor selects action — pure exploitation, no noise
            G      = self._build_state(S_curr, I, np.array(history))
            with torch.no_grad():
                action = self.actor(G).item()  # in [I_min, I_max]

            # Clip to inventory bounds
            I_new   = np.clip(action, self.config.I_min, self.config.I_max)
            delta_I = I_new - I  # actual trade needed

            # r = I_{t+1}*(S_{t+1} - S_t) - lambda*|delta_I|  (Eq. 4: inventory after action)
            r = I_new * diff_S - self.config.transaction_cost * abs(delta_I)

            rewards.append(r)
            inventories.append(I)
            price_diffs.append(diff_S)

            I = I_new
            history.append(S_next)

        rewards     = np.array(rewards)
        inventories = np.array(inventories)
        price_diffs = np.array(price_diffs)
        cumulative  = np.cumsum(rewards)

        # Hit Ratio Analysis
        # 1. Overall Hit Ratio: sign(I) matches sign(diff_S)
        # Note: sign is 0 if value is 0. We focus on non-zero cases.
        pos_move = price_diffs > 0
        neg_move = price_diffs < 0
        
        long_hit  = (inventories[pos_move] > 0).mean() if np.any(pos_move) else 0
        short_hit = (inventories[neg_move] < 0).mean() if np.any(neg_move) else 0
        
        # Overall directional accuracy: matches for any non-zero inventory
        active_mask = (inventories != 0)
        if np.any(active_mask):
            hit_ratio = (np.sign(inventories[active_mask]) == np.sign(price_diffs[active_mask])).mean()
        else:
            hit_ratio = 0

        return {
            "cumulative_reward" : cumulative[-1],
            "cumulative_curve"  : cumulative,
            "rewards"           : rewards,
            "inventories"       : inventories,
            "long_hit_ratio"    : long_hit,
            "short_hit_ratio"   : short_hit,
            "hit_ratio"         : hit_ratio
        }

    # ── Run all episodes ─────────────────────────────────────────────────────

    def run_all(self, episodes: list) -> dict:
        """Run the actor on a list of OU episodes and aggregate stats."""
        results      = [self.run_episode(ep) for ep in episodes]
        cum_rewards  = [r["cumulative_reward"] for r in results]
        long_hits    = [r["long_hit_ratio"] for r in results]
        short_hits   = [r["short_hit_ratio"] for r in results]
        overall_hits = [r["hit_ratio"] for r in results]

        print(f"Episodes: {len(results)} | "
              f"Mean cumulative reward: {np.mean(cum_rewards):.4f} "
              f"+/- {np.std(cum_rewards):.4f}")
        
        print(f"  Long Hit Ratio (I>0 when S↑):  {np.mean(long_hits):.2%}")
        print(f"  Short Hit Ratio (I<0 when S↓): {np.mean(short_hits):.2%}")
        print(f"  Overall Directional Hit:      {np.mean(overall_hits):.2%}")

        return {
            "mean_cumulative_reward" : np.mean(cum_rewards),
            "std_cumulative_reward"  : np.std(cum_rewards),
            "mean_long_hit_ratio"    : np.mean(long_hits),
            "mean_short_hit_ratio"   : np.mean(short_hits),
            "mean_hit_ratio"         : np.mean(overall_hits),
            "all_results"            : results,
        }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import argparse

    p = argparse.ArgumentParser(description="Evaluate a trained DDPG agent on held-out test episodes")
    p.add_argument("--mode", default="reg", choices=["reg", "prob"])
    p.add_argument("--case", type=int,  default=1, choices=[1, 2, 3])
    p.add_argument("--data_dir", default=None,
                   help="Path to test episodes (default: replay_buffer/data/<case>_test/)")
    p.add_argument("--num_episodes", type=int, default=None,
                   help="Number of episodes to evaluate (default: all)")
    a = p.parse_args()

    _dirs = {1: "theta_MK", 2: "theta_kappa_MK", 3: "theta_kappa_sigma_MK"}
    data_dir   = a.data_dir or f"replay_buffer/data/{_dirs[a.case]}_test"
    model_path = f"models/checkpoints/ddpg_{a.mode}_case{a.case}_best.pth"
    config     = SimulationConfig()

    files    = glob.glob(os.path.join(data_dir, "episode_*.csv"))
    if a.num_episodes is not None:
        files = files[:a.num_episodes]
    episodes = [pd.read_csv(f)['S'].values for f in sorted(files)]
    print(f"Loaded {len(episodes)} test episodes from {data_dir}")

    env   = TradingEnvironment(config, a.mode, model_path)
    stats = env.run_all(episodes)
    print(f"\nFinal mean cumulative reward: {stats['mean_cumulative_reward']:.4f}")