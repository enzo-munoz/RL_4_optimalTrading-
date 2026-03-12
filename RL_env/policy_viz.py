"""
Policy Visualization — 3D + 2D heatmaps for the trained DDPG actor.

Sweeps the state space (S_t, I_t) for each theta regime and plots
the actor's action output to diagnose short/long bias.

Usage:
    .venv/Scripts/python RL_env/policy_viz.py --mode prob --case 1
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.constants import (
    SimulationConfig,
    GRU_HIDDEN_DIM, GRU_HIDDEN_LAYERS,
    CASE_ARCHITECTURES,
)
from models.gru import GRUNet
from models.actor import Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ou_history(theta: float, W: int, config: SimulationConfig,
                    seed: int | None = None) -> np.ndarray:
    """
    Generate W OU price steps with theta fixed (Case 1: kappa=5, sigma=0.2).
    Returns array of shape (W,).
    """
    rng = np.random.default_rng(seed)
    kappa = 5.0   # Case-1 fixed value
    sigma = 0.2   # Case-1 fixed value
    S = config.mu_inv
    history = []
    for _ in range(W):
        dW = rng.standard_normal() * np.sqrt(config.dt)
        S += kappa * (theta - S) * config.dt + sigma * dW
        history.append(S)
    return np.array(history, dtype=np.float32)


def gru_phi(gru: GRUNet, history: np.ndarray) -> torch.Tensor:
    """Run frozen GRU on a (W,) price history → phi tensor shape (gru_out_dim,)."""
    x = torch.FloatTensor(history).unsqueeze(0).unsqueeze(-1).to(DEVICE)  # (1,W,1)
    with torch.no_grad():
        return gru(x).squeeze(0)   # (gru_out_dim,)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_action_grid(
    actor: Actor,
    phi: torch.Tensor,
    S_grid: np.ndarray,
    I_grid: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """
    Query actor for every (S, I) pair with a fixed GRU embedding phi.
    Returns actions array of shape (len(I_grid), len(S_grid)).
    Rows = I (y-axis), columns = S (x-axis) — matches imshow convention.
    """
    actions = np.empty((len(I_grid), len(S_grid)), dtype=np.float32)
    phi_batch = phi.unsqueeze(0)  # (1, gru_out_dim) — reused for every grid point

    # Build all states at once for speed
    S_norm = (S_grid - config.mu_inv) / 0.5
    I_norm = 2.0 * (I_grid - config.I_min) / (config.I_max - config.I_min) - 1.0

    # Vectorise over S, loop over I (40 iterations)
    with torch.no_grad():
        for j, i_n in enumerate(I_norm):
            s_t = torch.FloatTensor(S_norm).to(DEVICE)          # (40,)
            i_t = torch.full((len(S_norm),), i_n, device=DEVICE) # (40,)
            phi_t = phi_batch.expand(len(S_norm), -1)            # (40, gru_out_dim)
            G = torch.stack([s_t, i_t], dim=1)                   # (40, 2)
            G = torch.cat([G, phi_t], dim=1)                     # (40, state_dim)
            a = actor(G).squeeze(-1).cpu().numpy()               # (40,)
            actions[j] = a

    return actions


def main():
    p = argparse.ArgumentParser(description="Visualize DDPG actor policy in 3D + heatmaps")
    p.add_argument("--mode", default="prob", choices=["reg", "prob"])
    p.add_argument("--case", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--model_path", default=None,
                   help="Override checkpoint path (default: models/checkpoints/ddpg_{mode}_case{case}_best.pth)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for OU history generation")
    p.add_argument("--n_histories", type=int, default=20,
                   help="Number of random OU histories averaged per regime (default 20)")
    p.add_argument("--out_dir", default="RL_env",
                   help="Output directory for PNGs (default: RL_env/)")
    args = p.parse_args()

    config = SimulationConfig()
    W      = config.lookback_window_prob if args.mode == "prob" else config.lookback_window_reg

    model_path = args.model_path or f"models/checkpoints/ddpg_{args.mode}_case{args.case}_best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # ── Load models ───────────────────────────────────────────────────────────
    gru_out_dim    = 1 if args.mode == "reg" else len(config.theta_values)
    hidden_layers  = CASE_ARCHITECTURES[args.case]
    state_dim      = 2 + gru_out_dim

    gru   = GRUNet(1, GRU_HIDDEN_DIM, GRU_HIDDEN_LAYERS,
                   output_size=gru_out_dim, head_type=args.mode).to(DEVICE)
    actor = Actor(state_dim, 1, hidden_layers=hidden_layers).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)
    gru.load_state_dict(ckpt['gru'])
    actor.load_state_dict(ckpt['actor'])
    gru.eval(); actor.eval()
    print(f"Loaded checkpoint: {model_path}")
    print(f"  mode={args.mode}, case={args.case}, state_dim={state_dim}, gru_out_dim={gru_out_dim}")

    # ── Representative GRU embeddings per theta regime ────────────────────────
    theta_values = config.theta_values   # [0.9, 1.0, 1.1]
    phis = []
    for i, theta in enumerate(theta_values):
        # Average over n_histories random OU histories for stability
        phi_sum = None
        for k in range(args.n_histories):
            hist = make_ou_history(theta, W, config, seed=args.seed * 100 + i * 10 + k)
            p_k  = gru_phi(gru, hist)
            phi_sum = p_k if phi_sum is None else phi_sum + p_k
        phis.append(phi_sum / args.n_histories)

    if args.mode == "prob":
        for i, (theta, phi) in enumerate(zip(theta_values, phis)):
            probs = phi.cpu().numpy()
            print(f"  θ={theta}: GRU softmax → [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")
    else:
        for i, (theta, phi) in enumerate(zip(theta_values, phis)):
            print(f"  θ={theta}: GRU prediction → {phi.cpu().item():.4f}")

    # ── Grid sweep ────────────────────────────────────────────────────────────
    S_grid = np.linspace(0.6, 1.4, 40)
    I_grid = np.linspace(-10, 10, 40)

    action_grids = []
    for theta, phi in zip(theta_values, phis):
        grid = build_action_grid(actor, phi, S_grid, I_grid, config)
        action_grids.append(grid)
        print(f"  θ={theta}: action range [{grid.min():.2f}, {grid.max():.2f}]  "
              f"mean={grid.mean():.3f}  std={grid.std():.3f}")

    vmin, vmax = -10.0, 10.0

    # ── Figure 1 — 3D scatter ─────────────────────────────────────────────────
    fig3d = plt.figure(figsize=(10, 7))
    ax    = fig3d.add_subplot(111, projection='3d')

    SS, II = np.meshgrid(S_grid, I_grid)   # both shape (40, 40)
    for theta, grid in zip(theta_values, action_grids):
        z_plane = np.full_like(SS, theta)
        sc = ax.scatter(
            SS.ravel(), II.ravel(), z_plane.ravel(),
            c=grid.ravel(), cmap='RdBu', vmin=vmin, vmax=vmax,
            s=8, alpha=0.7, depthshade=False,
        )

    fig3d.colorbar(sc, ax=ax, label="Action (inventory target)", pad=0.1, shrink=0.6)
    ax.set_xlabel("S_t (price)")
    ax.set_ylabel("I_t (inventory)")
    ax.set_zlabel("θ regime")
    ax.set_zticks(theta_values)
    ax.set_title(f"Actor Policy — mode={args.mode}, case={args.case}\n"
                 f"Blue=short, Red=long")

    out_3d = os.path.join(args.out_dir, "policy_viz_3d.png")
    fig3d.savefig(out_3d, dpi=150, bbox_inches="tight")
    plt.close(fig3d)
    print(f"Saved: {out_3d}")

    # ── Figure 2 — 2D heatmaps ────────────────────────────────────────────────
    fig2d, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig2d.suptitle(f"Actor Policy Heatmaps — mode={args.mode}, case={args.case}\n"
                   f"X=S_t, Y=I_t   |   Blue=short, Red=long", fontsize=12)

    for ax2, theta, grid in zip(axes, theta_values, action_grids):
        im = ax2.imshow(
            grid,
            extent=[S_grid[0], S_grid[-1], I_grid[0], I_grid[-1]],
            origin='lower', aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax,
        )
        ax2.set_title(f"θ = {theta}")
        ax2.set_xlabel("S_t (price)")
        if ax2 is axes[0]:
            ax2.set_ylabel("I_t (inventory)")
        # Mark zero-action contour
        ax2.contour(S_grid, I_grid, grid, levels=[0.0], colors='k', linewidths=1.0)

    fig2d.colorbar(im, ax=axes, label="Action (inventory target)", shrink=0.8)
    out_2d = os.path.join(args.out_dir, "policy_viz_heatmaps.png")
    fig2d.savefig(out_2d, dpi=150, bbox_inches="tight")
    plt.close(fig2d)
    print(f"Saved: {out_2d}")


if __name__ == "__main__":
    main()
