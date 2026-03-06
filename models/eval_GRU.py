import argparse
import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gru import GRUNet
from models.gru_utils import load_episode_data, create_sequences, prepare_dataloaders
from lib.constants import GRU_HIDDEN_DIM, LOOKBACK_WINDOW, GRU_HIDDEN_LAYERS, GRU_OUTPUT_SIZE, SimulationConfig

def evaluate_episode(episode_path, model_path, type="prob"):
    """
    Evaluates the GRU model on a single episode and plots predictions.
    """
    print(f"Loading data from {episode_path}...")
    try:
        df = load_episode_data(episode_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    if type == "prob":
        target_col = 'theta'
        output_size = len(SimulationConfig.theta_values)
        task_type = "prob"
    else:
        target_col = 'S'
        output_size = GRU_OUTPUT_SIZE
        task_type = "reg"
    
    input_cols = ['S']
    
    X, y = create_sequences(df, input_cols=input_cols, target_col=target_col, lookback=LOOKBACK_WINDOW)
    
    if len(X) == 0:
        print("Not enough data.")
        return

    # Split to match training (assumes same split logic)
    test_size = 0.2
    # We don't need loaders for full evaluation plotting, but we need the split indices
    # Re-use prepare_dataloaders to get the tensors directly
    _, _, (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor) = prepare_dataloaders(
        X, y, batch_size=1, test_size=test_size, shuffle_train=False, task_type=task_type
    )
    
    # Load Model
    input_size = len(input_cols)
    hidden_size = GRU_HIDDEN_DIM
    num_layers = GRU_HIDDEN_LAYERS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUNet(input_size, hidden_size, num_layers, output_size=output_size, head_type=type)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Running predictions...")
    with torch.no_grad():
        if type == "prob":
            # Returns probabilities due to Softmax in head
            train_outputs = model(X_train_tensor.to(device))
            test_outputs = model(X_test_tensor.to(device))
            
            # Get class indices
            train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
        else:
            train_preds = model(X_train_tensor.to(device)).cpu().numpy()
            test_preds = model(X_test_tensor.to(device)).cpu().numpy()
        
    # Plotting
    # We need to map predictions back to the time axis.
    
    # Time axis
    t = df['t'].values
    start_idx = LOOKBACK_WINDOW
    
    train_len = len(train_preds)
    test_len = len(test_preds)
    
    # Indices in original dataframe
    train_indices = range(start_idx, start_idx + train_len)
    test_indices = range(start_idx + train_len, start_idx + train_len + test_len)
    
    t_train = t[train_indices]
    t_test = t[test_indices]
    
    y_train_true = y_train_tensor.numpy()
    y_test_true = y_test_tensor.numpy()
    
    plt.figure(figsize=(12, 6))
    
    if type == "prob":
        # Plot true theta values (need to map back if we want values, but indices are fine for comparison)
        # Note: y_train_true are indices now because create_sequences handles it.
        # But df[target_col] has original float values.
        # Let's plot the indices from our tensors to match predictions.
        
        plt.plot(t_train, y_train_true, 'b--', label='True Train (Index)', alpha=0.5)
        plt.plot(t_train, train_preds, 'bx', label='Pred Train (Index)')
        
        plt.plot(t_test, y_test_true, 'r--', label='True Test (Index)', alpha=0.5)
        plt.plot(t_test, test_preds, 'rx', label='Pred Test (Index)')
        
        # Add y-ticks for classes
        plt.yticks(range(len(SimulationConfig.theta_values)), SimulationConfig.theta_values)
        plt.ylabel(f"{target_col} (Class)")
    else:
        # Plot True Data (Background)
        plt.plot(t, df[target_col], color='lightgray', label='Full True Signal', linewidth=1)
        
        # Plot Train Predictions vs True
        plt.plot(t_train, y_train_true, 'b--', label='True Train', alpha=0.7)
        plt.plot(t_train, train_preds, 'b-', label='Pred Train')
        
        # Plot Test Predictions vs True
        plt.plot(t_test, y_test_true, 'r--', label='True Test', alpha=0.7)
        plt.plot(t_test, test_preds, 'r-', label='Pred Test')
        plt.ylabel(target_col)
    
    plt.axvline(x=t[start_idx + train_len], color='k', linestyle=':', label='Train/Test Split')
    
    plt.xlabel('Time (t)')
    plt.title(f'GRU Predictions vs True Signal ({target_col})\nEpisode: {os.path.basename(episode_path)}')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f"evaluation_plot_{type}.png"
    plt.savefig(plot_filename)
    print(f"Evaluation plot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GRU on a single episode")
    parser.add_argument("--episode", type=str, required=True, help="Path to the episode CSV file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--type", type=str, default="prob", help="Type of architecture (prob, reg, hid)")
    
    args = parser.parse_args()
    
    evaluate_episode(args.episode, args.model, args.type)
