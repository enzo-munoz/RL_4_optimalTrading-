import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.constants import SimulationConfig

file_path = sys.argv[1] if len(sys.argv) > 1 else "replay_buffer/data/theta_MK/episode_1.csv"

df = pd.read_csv(file_path)

thetas = SimulationConfig.theta_values
colors = plt.cm.viridis([i / (len(thetas) - 1) for i in range(len(thetas))])
color_map = dict(zip(thetas, colors))

fig, ax = plt.subplots(figsize=(12, 7))

for theta in thetas:
    subset = df[df['theta'] == theta].reset_index(drop=True)
    # Détecte les sauts dans t (début d'un nouvel épisode)
    breaks = subset.index[subset['t'].diff() > 1.5 * SimulationConfig.dt].tolist()
    breaks = [0] + breaks + [len(subset)]
    
    for i in range(len(breaks) - 1):
        segment = subset.iloc[breaks[i]:breaks[i+1]]
        ax.plot(segment['t'], segment['S'], color=color_map[theta], linewidth=0.8)

# Légende discrète
handles = [plt.Line2D([0], [0], color=color_map[t], linewidth=2, label=f'θ={t:.3f}') for t in thetas]
ax.legend(handles=handles, title='Theta', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

ax.set_xlabel("t")
ax.set_ylabel("S")
ax.set_title("S en fonction de t")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot.png", dpi=150)
plt.show()