import argparse
import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import math

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gru import GRUNet
from models.gru_utils import load_episode_data, create_sequences, prepare_dataloaders
from lib.constants import GRU_HIDDEN_DIM, LOOKBACK_WINDOW_PROB, LOOKBACK_WINDOW_REG, GRU_HIDDEN_LAYERS, GRU_OUTPUT_SIZE, SimulationConfig
from scipy.stats import pearsonr
import warnings

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
        lookback = LOOKBACK_WINDOW_PROB
    else:
        target_col = 'S'
        output_size = GRU_OUTPUT_SIZE
        task_type = "reg"
        lookback = LOOKBACK_WINDOW_REG

    input_cols = ['S']

    X, y = create_sequences(df, input_cols=input_cols, target_col=target_col, lookback=lookback)
    
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
    
    # Calculate Metrics
    y_train_true = y_train_tensor.numpy()
    y_test_true = y_test_tensor.numpy()
    
    metrics_str = ""
    if type == "prob":
        train_acc = accuracy_score(y_train_true, train_preds)
        test_acc = accuracy_score(y_test_true, test_preds)
        train_f1 = f1_score(y_train_true, train_preds, average='weighted')
        test_f1 = f1_score(y_test_true, test_preds, average='weighted')
        print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        metrics_str = f"\nTrain Acc: {train_acc:.2f}, F1: {train_f1:.2f} | Test Acc: {test_acc:.2f}, F1: {test_f1:.2f}"
    else:
        train_rmse = math.sqrt(mean_squared_error(y_train_true, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test_true, test_preds))

        # Diagnostic statistics
        baseline_pred = np.mean(y_train_true)
        baseline_rmse = math.sqrt(np.mean((y_test_true - baseline_pred) ** 2))
        r2 = 1.0 - (test_rmse ** 2) / (baseline_rmse ** 2) if baseline_rmse > 0 else float('nan')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(test_preds) > 1 and np.std(test_preds) > 0:
                corr, p_val = pearsonr(test_preds.flatten(), y_test_true.flatten())
            else:
                corr, p_val = float('nan'), float('nan')

        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Baseline RMSE (predict train mean): {baseline_rmse:.4f}")
        print(f"R² score: {r2:.4f}")
        print(f"Pearson correlation: {corr:.4f} (p={p_val:.2e})")
        print(f"\n--- Prediction stats ---")
        print(f"  Preds  — mean: {np.mean(test_preds):.6f}, std: {np.std(test_preds):.6f}, min: {np.min(test_preds):.6f}, max: {np.max(test_preds):.6f}")
        print(f"  True   — mean: {np.mean(y_test_true):.6f}, std: {np.std(y_test_true):.6f}, min: {np.min(y_test_true):.6f}, max: {np.max(y_test_true):.6f}")

        collapsed = np.std(test_preds) < 0.01 * np.std(y_test_true)
        if collapsed:
            print(f"\n  *** VERDICT: Model has COLLAPSED (pred std {np.std(test_preds):.6f} << true std {np.std(y_test_true):.6f}) ***")
        else:
            print(f"\n  Verdict: Model appears to produce varied predictions.")

        metrics_str = f"\nTrain RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | R²: {r2:.4f}"
    
    # Time axis
    t = df['t'].values
    start_idx = lookback
    
    train_len = len(train_preds)
    test_len = len(test_preds)
    
    # Indices in original dataframe
    train_indices = range(start_idx, start_idx + train_len)
    test_indices = range(start_idx + train_len, start_idx + train_len + test_len)
    
    t_train = t[train_indices]
    t_test = t[test_indices]
    
    plt.figure(figsize=(12, 6))
    
    if type == "prob":
        # Plot true theta values (need to map back if we want values, but indices are fine for comparison)
        # Note: y_train_true are indices now because create_sequences handles it.
        # But df[target_col] has original float values.
        # Let's plot the indices from our tensors to match predictions.
        
        plt.plot(t_train, y_train_true, 'b--', label='True Train (Index)', alpha=0.5)
        plt.plot(t_train, train_preds, 'gx', label='Pred Train (Index)')
        
        plt.plot(t_test, y_test_true, 'r--', label='True Test (Index)', alpha=0.5)
        plt.plot(t_test, test_preds, 'mx', label='Pred Test (Index)')
        
        # Add y-ticks for classes
        plt.yticks(range(len(SimulationConfig.theta_values)), SimulationConfig.theta_values)
        plt.ylabel(f"{target_col} (Class)")
    else:
        # Plot True Data (Background)
        plt.plot(t, df[target_col], color='lightgray', label='Full True Signal', linewidth=1)
        
        # Plot Train Predictions vs True
        plt.plot(t_train, y_train_true, 'b--', label='True Train', alpha=0.7)
        plt.plot(t_train, train_preds, 'g-', label='Pred Train')
        
        # Plot Test Predictions vs True
        plt.plot(t_test, y_test_true, 'r--', label='True Test', alpha=0.7)
        plt.plot(t_test, test_preds, 'g-', label='Pred Test')
        plt.ylabel(target_col)
    
    plt.axvline(x=t[start_idx + train_len], color='k', linestyle=':', label='Train/Test Split')
    
    plt.xlabel('Time (t)')
    plt.title(f'GRU Predictions vs True Signal ({target_col})\nEpisode: {os.path.basename(episode_path)}{metrics_str}')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists("eval_plots"):
        os.makedirs("eval_plots")
    
    plot_filename = f"evaluation_plot_{type}.png"
    plt.savefig(os.path.join("eval_plots", plot_filename))
    print(f"Evaluation plot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GRU on a single episode")
    parser.add_argument("--episode", type=str, required=True, help="Path to the episode CSV file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--type", type=str, default="prob", help="Type of architecture (prob, reg)")
    
    args = parser.parse_args()
    
    evaluate_episode(args.episode, args.model, args.type)
