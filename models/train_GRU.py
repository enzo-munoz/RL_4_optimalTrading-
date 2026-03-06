import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gru import GRUNet
from models.gru_utils import load_episode_data, create_sequences, prepare_dataloaders
from lib.constants import GRU_HIDDEN_DIM, LOOKBACK_WINDOW, GRU_HIDDEN_LAYERS, GRU_OUTPUT_SIZE, SimulationConfig

def train_all_episodes(data_dir, output_dir="models/checkpoints", epochs=50, batch_size=32, lr=0.001, type="prob"):
    """
    Trains the GRU model on all episodes in the directory.
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    if not type in ["reg", "hid", "prob"]: 
        print(f"Architecture not known : {type}")
        return
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_files = all_files[:50]
    print(f"Found {len(all_files)} episodes in {data_dir}")
    
    if not all_files:
        print("No csv files found.")
        return

    all_X = []
    all_y = []
    
    # Updated input columns: Only 'S' is used
    input_cols = ['S']
    
    if type == "prob":
        target_col = 'theta'
        task_type = "prob"
        # Output size is number of classes
        output_size = len(SimulationConfig.theta_values)
        criterion = nn.CrossEntropyLoss()
    else: 
        target_col = "S"
        task_type = "reg"
        output_size = GRU_OUTPUT_SIZE
        criterion = nn.MSELoss()

    print(f"Training for type: {type}, Target: {target_col}, Output Size: {output_size}")
    print("Loading and processing episodes...")
    for file_path in all_files:
        try:
            df = load_episode_data(file_path)
            X, y = create_sequences(df, input_cols=input_cols, target_col=target_col, lookback=LOOKBACK_WINDOW)
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if not all_X:
        print("No valid data sequences created.")
        return
        
    # Concatenate all sequences
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"Total samples across all episodes: {len(X)}")
    
    # Split train/test (last 20% for testing/validation)
    test_size = 0.2
    train_loader, test_loader, _, _ = prepare_dataloaders(X, y, batch_size=batch_size, test_size=test_size, task_type=task_type)
    
    # Model Setup
    input_size = len(input_cols) # Should be 1
    hidden_size = GRU_HIDDEN_DIM
    num_layers = GRU_HIDDEN_LAYERS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GRUNet(input_size, hidden_size, num_layers, output_size=output_size, head_type=type)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_model_state = None
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        
        if len(test_loader) > 0:
            val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            # Save best model
            torch.save(best_model_state, os.path.join(output_dir, f"best_gru_model_{type}.pth"))
            
    print("Training complete.")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU on all episodes")
    # Changed argument from --episode to --data_dir
    parser.add_argument("--data_dir", type=str, default="replay_buffer/data/theta_MK", help="Directory containing episode CSV files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--out_dir", type=str, default="models/checkpoints", help="Output directory for model and plots")
    parser.add_argument("--type", type=str, default="prob", help="Type of architecture")

    args = parser.parse_args()
    
    train_all_episodes(data_dir=args.data_dir, output_dir=args.out_dir, epochs=args.epochs, type=args.type)
