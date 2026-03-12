import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gru import GRUNet
from models.gru_utils import load_episode_data, create_sequences, prepare_dataloaders
from lib.constants import (
    GRU_HIDDEN_DIM, GRU_HIDDEN_LAYERS, GRU_OUTPUT_SIZE,
    LOOKBACK_WINDOW_PROB, LOOKBACK_WINDOW_REG,
    SimulationConfig,
)
from lib.win_adam import WinAdam


def train_all_episodes(
    data_dir,
    output_dir="models/checkpoints",
    epochs=50,
    batch_size=256,
    lr=0.001,
    type="prob",
    max_episodes=None,
):
    """
    Trains the GRU model on all episodes in data_dir.

    Optimizer: W-ADAM with window=W (10 for prob, 50 for reg) — Table 2.
    Lookback:  W=10 for prob-DDPG, W=50 for reg-DDPG            — Table 2.
    No episode cap: uses all available CSV files                 — Table 2 (10 000 episodes).
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    if type not in ["reg", "hid", "prob"]:
        print(f"Architecture not known: {type}")
        return

    # Select lookback window based on mode (Table 2)
    lookback = LOOKBACK_WINDOW_PROB if type == "prob" else LOOKBACK_WINDOW_REG

    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if max_episodes is not None:
        all_files = all_files[:max_episodes]
    print(f"Found {len(all_files)} episodes in {data_dir}  (lookback W={lookback})")

    if not all_files:
        print("No csv files found.")
        return

    input_cols = ['S']

    if type == "prob":
        target_col  = 'theta'
        task_type   = "prob"
        output_size = len(SimulationConfig.theta_values)
        criterion   = nn.CrossEntropyLoss()
    else:
        target_col  = "S"
        task_type   = "reg"
        output_size = GRU_OUTPUT_SIZE
        criterion   = nn.MSELoss()

    print(f"Training for type: {type}, Target: {target_col}, Output Size: {output_size}")
    print("Loading and processing episodes...")

    all_X, all_y = [], []
    for file_path in all_files:
        try:
            df = load_episode_data(file_path)
            X, y = create_sequences(
                df, input_cols=input_cols, target_col=target_col, lookback=lookback
            )
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_X:
        print("No valid data sequences created.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Total samples: {len(X)}")

    train_loader, test_loader, _, _ = prepare_dataloaders(
        X, y, batch_size=batch_size, test_size=0.2, task_type=task_type
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GRUNet(
        input_size=len(input_cols),
        hidden_size=GRU_HIDDEN_DIM,
        num_layers=GRU_HIDDEN_LAYERS,
        output_size=output_size,
        head_type=type,
    ).to(device)

    # W-ADAM optimizer with window matching the lookback window (Table 2)
    optimizer = WinAdam(model.parameters(), lr=lr, window=lookback)

    os.makedirs(output_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = np.inf

    n_batches = len(train_loader)
    print(f"Starting training... ({epochs} epochs, {n_batches} batches/epoch)")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Intra-epoch progress every 10% of batches
            if n_batches >= 10 and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                pct = 100 * (batch_idx + 1) / n_batches
                avg = running_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{epochs} [{pct:3.0f}%] batch {batch_idx+1}/{n_batches}  running loss: {avg:.4f}", end="\r")

        train_loss = running_loss / n_batches
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_loss += criterion(model(x_val), y_val).item()
        if len(test_loader) > 0:
            val_loss /= len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}" + " " * 30)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_gru_model_{type}.pth"))
            print(f"  → New best model saved (val loss: {best_val_loss:.4f})")

    print("Training complete.")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.title("GRU Training and Validation Loss")
    plt.legend(); plt.grid(True)
    plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU on all episodes")
    parser.add_argument("--data_dir", type=str, default="replay_buffer/data/theta_MK")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--out_dir",      type=str,   default="models/checkpoints")
    parser.add_argument("--type",         type=str,   default="prob", choices=["prob", "reg", "hid"])
    parser.add_argument("--max_episodes", type=int,   default=None,  help="Cap number of episodes (default: all)")
    args = parser.parse_args()

    train_all_episodes(
        data_dir=args.data_dir,
        output_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        type=args.type,
        max_episodes=args.max_episodes,
    )
