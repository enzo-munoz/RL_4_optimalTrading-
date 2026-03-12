import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import deque
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.constants import SimulationConfig
from trading_env import TradingEnvironment, DEVICE

def run_detailed_episode(env: TradingEnvironment, S_series: np.ndarray):
    """
    Runs a single episode and records all inputs, actions, and rewards.
    """
    T = len(S_series)
    history = deque(S_series[:env.W], maxlen=env.W)
    I = 0.0

    data_log = []

    for t in range(env.W, T - 1):
        S_curr = S_series[t]
        S_next = S_series[t + 1]
        
        # Build state and extract components for logging
        S_norm = (S_curr - env.config.mu_inv) / 0.5
        I_norm = 2 * (I - env.config.I_min) / (env.config.I_max - env.config.I_min) - 1
        
        hist_t = torch.FloatTensor(np.array(history)).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            phi = env.gru(hist_t).squeeze(0)
            
        # Reconstruct state G as in trading_env.py
        G = torch.cat([
            torch.tensor([S_norm, I_norm], dtype=torch.float32, device=DEVICE), phi
        ]).unsqueeze(0)
        
        # Get action (target inventory I)
        with torch.no_grad():
            action = env.actor(G).item()
            
        # Clip and calculate actual delta_I
        I_new = np.clip(action, env.config.I_min, env.config.I_max)
        delta_I = I_new - I
        
        # Calculate reward and price diff
        diff_S = S_next - S_curr
        reward = I * diff_S - env.config.transaction_cost * abs(delta_I)
        
        # Log everything
        log_entry = {
            'step': t,
            'S_curr': S_curr,
            'S_next': S_next,
            'diff_S': diff_S,
            'I': I,
            'S_norm': S_norm,
            'I_norm': I_norm,
            'action': action,
            'delta_I': delta_I,
            'reward': reward
        }
        
        # Add GRU outputs to log
        for i, p in enumerate(phi.cpu().numpy()):
            log_entry[f'phi_{i}'] = p
            
        data_log.append(log_entry)
        
        # Update for next step
        I = I_new
        history.append(S_next)
        
    df = pd.DataFrame(data_log)
    df['cumulative_reward'] = df['reward'].cumsum()

    # Hit ratio stats
    pos_move  = df['diff_S'] > 0
    neg_move  = df['diff_S'] < 0
    long_hit  = (df.loc[pos_move, 'I'] > 0).mean() if pos_move.any() else 0
    short_hit = (df.loc[neg_move, 'I'] < 0).mean() if neg_move.any() else 0
    active    = df['I'] != 0
    overall   = (np.sign(df.loc[active, 'I']) == np.sign(df.loc[active, 'diff_S'])).mean() if active.any() else 0

    print(f"\n--- Hit Ratio ---")
    print(f"  Long  Hit (I>0 when S up):   {long_hit:.2%}")
    print(f"  Short Hit (I<0 when S down): {short_hit:.2%}")
    print(f"  Overall Directional Hit:  {overall:.2%}")
    print(f"  Cumulative Reward:        {df['cumulative_reward'].iloc[-1]:.4f}")

    return df

def plot_episode_results(df: pd.DataFrame, output_path: str):
    """
    Plots cumulative reward, actions, and price differences on the same figure.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Cumulative Reward
    ax1.plot(df['step'], df['cumulative_reward'], color='blue', label='Cumulative Reward')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Episode Analysis: Rewards, Actions and Price Diffs')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Actions taken
    ax2.step(df['step'], df['action'], where='post', color='green', label='Action (Target I)')
    ax2.set_ylabel('Target Inventory')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Price Difference (St+1 - St)
    ax3.bar(df['step'], df['diff_S'], color='red', alpha=0.5, label='S_{t+1} - S_t')
    ax3.set_ylabel('Price Diff')
    ax3.set_xlabel('Step')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    config = SimulationConfig()
    mode = "prob"
    case = 1
    model_path = f"models/checkpoints/ddpg_{mode}_case{case}_best.pth"
    data_dir = "replay_buffer/data/theta_MK"
    
    # Load one episode
    files = glob.glob(os.path.join(data_dir, "**", "episode_*.csv"), recursive=True)
    if not files:
        print(f"No episodes found in {data_dir}")
        sys.exit(1)
    
    # Use episode_1 for analysis
    ep_file = sorted(files)[2]
    print(f"Loading episode from: {ep_file}")
    S_series = pd.read_csv(ep_file)['S'].values
    
    # Initialize environment
    env = TradingEnvironment(config, mode, model_path)
    
    # Run detailed analysis
    print("Running detailed episode analysis...")
    results_df = run_detailed_episode(env, S_series)
    
    # Save to CSV
    csv_output = "RL_env/episode_analysis.csv"
    results_df.to_csv(csv_output, index=False)
    print(f"Analysis data saved to {csv_output}")
    
    # Plot results
    plot_output = "RL_env/episode_analysis.png"
    plot_episode_results(results_df, plot_output)
