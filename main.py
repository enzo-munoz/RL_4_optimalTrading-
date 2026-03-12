"""
DDPG for Optimal Inventory Trading
Modes: 'reg' (GRU predicts S_{t+1}), 'prob' (GRU predicts theta distribution)

No replay buffer: at each training iteration fresh batches are generated
on-the-fly by sampling S_0 ~ N(µ_inv, 3·σ_inv) and I ~ U[I_min, I_max],
then simulating W+1 OU steps — consistent with paper Section 3.2.

Reward: r = I_{t+1} * (S_{t+1} - S_t) - λ|ΔI|   (Eq. 4, inventory after action)
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.constants import SimulationConfig, GRU_HIDDEN_DIM, GRU_HIDDEN_LAYERS
from models.gru import GRUNet
from models.actor import Actor
from models.critic import Critic
from OU.simulate_OU import OUProcess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Online Batch Generator ────────────────────────────────────────────────────

class OnlineBatchGenerator:
    """
    Generates a fresh batch of (S_hist, S_curr, S_next, I) on every call.

    From paper Section 3.2:
      "we choose not to keep memory of the states that the agent has visited
       during training. Rather, we feed randomly generated initial states."

    Per sample:
      S_0 ~ N(µ_inv, 3·σ_inv)   initial price
      I   ~ U[I_min, I_max]     random inventory
      Then W+1 OU steps → W prices of history, S_curr, S_next.
    """

    def __init__(self, config: SimulationConfig, case: int, W: int):
        self.config = config
        self.W      = W
        self.ou     = OUProcess(config, case=case)

    def get_batch(self, B: int):
        S_hist = np.zeros((B, self.W))
        S_curr = np.zeros(B)
        S_next = np.zeros(B)

        for b in range(B):
            S0 = np.random.normal(self.config.mu_inv, 3.0 * self.config.sigma_inv)
            self.ou.reset()
            self.ou.S = S0

            prices = [S0]
            for _ in range(self.W + 1):
                S, _, _, _ = self.ou.step()
                prices.append(S)

            S_hist[b] = prices[:self.W]
            S_curr[b] = prices[self.W]
            S_next[b] = prices[self.W + 1]

        I = np.random.uniform(self.config.I_min, self.config.I_max, B)
        return S_hist, S_curr, S_next, I


# ─── DDPG Agent ───────────────────────────────────────────────────────────────

class DDPGAgent:
    """
    DDPG with a frozen pre-trained GRU encoder.
    State  G = [S_norm, I_norm, GRU(S_hist)]          dim = 2 + gru_out_dim
    Action a = target inventory ∈ [I_min, I_max],     normalised to [-1,1] for critic
    """

    def __init__(
        self,
        config:   SimulationConfig,
        mode:     str,
        case:     int = 1,
        gru_path: str = None,
    ):
        self.config = config
        self.mode   = mode
        self.W      = (config.lookback_window_prob if mode == "prob"
                       else config.lookback_window_reg)

        # ── Frozen GRU encoder ──────────────────────────────────────────────
        gru_out_dim = 1 if mode == "reg" else len(config.theta_values)
        self.gru = GRUNet(
            input_size=1, hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_HIDDEN_LAYERS, output_size=gru_out_dim,
            head_type=mode,
        ).to(DEVICE)

        if gru_path and os.path.exists(gru_path):
            self.gru.load_state_dict(torch.load(gru_path, map_location=DEVICE))
            print(f"GRU loaded from {gru_path}")
        else:
            print("Warning: GRU using random weights.")

        for p in self.gru.parameters():
            p.requires_grad = False
        self.gru.eval()

        # ── Per-case Actor / Critic ─────────────────────────────────────────
        hidden_layers = config.case_architectures.get(case, config.case_architectures[1])
        state_dim     = 2 + gru_out_dim

        self.actor        = Actor(state_dim, 1, hidden_layers=hidden_layers).to(DEVICE)
        self.actor_target = Actor(state_dim, 1, hidden_layers=hidden_layers).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(state_dim, 1, hidden_layers=hidden_layers).to(DEVICE)
        self.critic_target = Critic(state_dim, 1, hidden_layers=hidden_layers).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=config.lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=config.lr)

        self.epsilon = 1.0

    # ── State construction ───────────────────────────────────────────────────

    def build_states(self, S: np.ndarray, I: np.ndarray, hist: np.ndarray) -> torch.Tensor:
        S_norm = torch.FloatTensor((S - self.config.mu_inv) / 0.5).unsqueeze(1).to(DEVICE)
        I_norm = torch.FloatTensor(
            2 * (I - self.config.I_min) / (self.config.I_max - self.config.I_min) - 1
        ).unsqueeze(1).to(DEVICE)
        hist_t = torch.FloatTensor(hist).unsqueeze(-1).to(DEVICE)   # (B, W, 1)
        with torch.no_grad():
            phi = self.gru(hist_t)
        return torch.cat([S_norm, I_norm, phi], dim=1)

    # ── Critic update ────────────────────────────────────────────────────────

    def update_critic(self, G, a, r, G_next):
        with torch.no_grad():
            a_next = self.actor_target(G_next) / self.config.MAX_ACTION
            y      = r + self.config.gamma * self.critic_target(G_next, a_next)
        loss = nn.MSELoss()(self.critic(G, a), y)
        self.opt_critic.zero_grad(); loss.backward(); self.opt_critic.step()
        self._soft_update(self.critic, self.critic_target)
        return loss.item()

    # ── Actor update ─────────────────────────────────────────────────────────

    def update_actor(self, G):
        loss = -self.critic(G, self.actor(G) / self.config.MAX_ACTION).mean()
        self.opt_actor.zero_grad(); loss.backward(); self.opt_actor.step()
        self._soft_update(self.actor, self.actor_target)
        return loss.item()

    # ── Soft target update ───────────────────────────────────────────────────

    def _soft_update(self, net, target):
        tau = self.config.tau
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

    def decay_epsilon(self, step: int):
        """Linearly decay ε from 1.0 to 0.01 over the first `a` iterations."""
        self.epsilon = max(0.01, 1.0 - step / self.config.a)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor':  self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'gru':    self.gru.state_dict(),
        }, path)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(mode: str, N: int, l_critic: int, l_actor: int, case: int):
    """
    Main training loop (no replay buffer):
      for m = 1..N:
        generate fresh batch on-the-fly
        select action a = π(G) + N(0, ε), clipped to [-1,1]
        compute reward r = I_next*(S_{t+1}-S_t) - λ|ΔI|   (Eq. 4)
        for ℓ: update critic
        for l: update actor
        decay ε using parameter a from config
    """
    config   = SimulationConfig()
    W        = config.lookback_window_prob if mode == "prob" else config.lookback_window_reg
    gru_path = f"models/checkpoints/best_gru_model_{mode}.pth"

    agent = DDPGAgent(config, mode, case=case, gru_path=gru_path)
    gen   = OnlineBatchGenerator(config, case=case, W=W)

    print(f"\n=== DDPG [{mode}] | Case {case} | N={N}, ℓ={l_critic}, l={l_actor} ===\n")

    best_reward = -np.inf
    best_path   = f"models/checkpoints/ddpg_{mode}_case{case}_best.pth"

    log_every = max(1, N // 100)   # ~1% of iterations
    t0 = time.time()

    for m in range(N):
        # ── Fresh batch (no replay buffer) ──────────────────────────────────
        S_hist, S_curr, S_next, I = gen.get_batch(config.batch_size)

        G = agent.build_states(S_curr, I, S_hist)
        G.detach_()

        # ── Select action: π(G) + N(0, ε), clipped to [-1,1] ───────────────
        agent.actor.eval()
        with torch.no_grad():
            a_norm = (agent.actor(G) / config.MAX_ACTION).cpu().numpy().flatten()
        agent.actor.train()
        a_noisy = np.clip(a_norm + np.random.normal(0, agent.epsilon, config.batch_size), -1, 1)

        # ── Reward: r = I_{t+1}*(S_{t+1}-S_t) - λ|ΔI|  (Eq. 4) ────────────
        I_next  = np.clip(a_noisy * config.MAX_ACTION, config.I_min, config.I_max)
        delta_I = I_next - I
        r       = I_next * (S_next - S_curr) - config.transaction_cost * np.abs(delta_I)

        # ── Next state ───────────────────────────────────────────────────────
        S_hist_next = np.concatenate([S_hist[:, 1:], S_next[:, None]], axis=1)
        G_next      = agent.build_states(S_next, I_next, S_hist_next)

        a_t = torch.FloatTensor(a_noisy).unsqueeze(1).to(DEVICE)
        r_t = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)

        # ── Update Critic (ℓ times) ──────────────────────────────────────────
        c_loss = sum(agent.update_critic(G, a_t, r_t, G_next) for _ in range(l_critic)) / l_critic

        # ── Update Actor (l times) ───────────────────────────────────────────
        a_loss = sum(agent.update_actor(G) for _ in range(l_actor)) / l_actor

        # ── Decay ε over first `a` iterations ───────────────────────────────
        agent.decay_epsilon(m)

        avg_reward = r.mean()
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(best_path)
            print(f"  → New best saved  r̄={best_reward:.4f}  (iter {m+1})")

        if (m + 1) % log_every == 0:
            elapsed  = time.time() - t0
            it_per_s = (m + 1) / elapsed
            eta_s    = (N - m - 1) / it_per_s
            eta_str  = f"{int(eta_s//60)}m{int(eta_s%60):02d}s"
            pct      = 100 * (m + 1) / N
            print(f"[{m+1:>5}/{N}] {pct:5.1f}%  C={c_loss:.4f}  A={a_loss:.4f}  "
                  f"r̄={avg_reward:.4f}  best={best_reward:.4f}  ε={agent.epsilon:.3f}  "
                  f"ETA {eta_str}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {int(elapsed//60)}m{int(elapsed%60):02d}s.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",     default="reg",  choices=["reg", "prob"])
    p.add_argument("--N",        type=int, default=1000)
    p.add_argument("--l_critic", type=int, default=1)
    p.add_argument("--l_actor",  type=int, default=1)
    p.add_argument("--case",     type=int, default=1, choices=[1, 2, 3],
                   help="OU market case: 1=θ only, 2=θ+κ, 3=θ+κ+σ")
    args = p.parse_args()
    train(args.mode, args.N, args.l_critic, args.l_actor, args.case)
