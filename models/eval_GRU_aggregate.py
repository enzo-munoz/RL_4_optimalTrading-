import argparse
import torch
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

def evaluate_episodes(data_dir, model_path, type="prob", num_episodes=50):
    """
    Evaluates the GRU model on multiple episodes and aggregates metrics.
    """
    
    if type == "prob":
        target_col = 'theta'
        output_size = len(SimulationConfig.theta_values)
        task_type = "prob"
        lookback = LOOKBACK_WINDOW_PROB
        print(f"Evaluating Probabilistic Model (F1 Score) on {num_episodes} episodes...")
    else:
        target_col = 'S'
        output_size = GRU_OUTPUT_SIZE
        task_type = "reg"
        lookback = LOOKBACK_WINDOW_REG
        print(f"Evaluating Regression Model (RMSE) on {num_episodes} episodes...")

    input_cols = ['S']
    
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

    total_f1 = 0
    total_rmse = 0
    total_baseline_rmse = 0
    total_r2 = 0
    total_pred_std = 0
    count = 0
    
    for i in range(num_episodes):
        episode_filename = f"episode_{i}.csv"
        episode_path = os.path.join(data_dir, episode_filename)
        
        if not os.path.exists(episode_path):
            print(f"Episode file not found: {episode_path}, skipping...")
            continue

        try:
            df = load_episode_data(episode_path)
        except Exception as e:
            print(f"Error loading {episode_path}: {e}")
            continue
        
        X, y = create_sequences(df, input_cols=input_cols, target_col=target_col, lookback=lookback)
        
        if len(X) == 0:
            print(f"Not enough data in {episode_filename}, skipping...")
            continue

        # Split to match training (assumes same split logic)
        test_size = 0.2
        # We don't need loaders for full evaluation plotting, but we need the split indices
        # Re-use prepare_dataloaders to get the tensors directly
        _, _, (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor) = prepare_dataloaders(
            X, y, batch_size=1, test_size=test_size, shuffle_train=False, task_type=task_type
        )
        
        # We will evaluate on the TEST set for aggregation
        
        with torch.no_grad():
            if type == "prob":
                # Returns probabilities due to Softmax in head
                test_outputs = model(X_test_tensor.to(device))
                # Get class indices
                test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            else:
                test_preds = model(X_test_tensor.to(device)).cpu().numpy()
            
        y_test_true = y_test_tensor.numpy()
        
        if type == "prob":
            f1 = f1_score(y_test_true, test_preds, average='weighted')
            total_f1 += f1
        else:
            rmse = math.sqrt(mean_squared_error(y_test_true, test_preds))
            total_rmse += rmse

            # Per-episode diagnostics
            y_train_true = y_train_tensor.numpy()
            baseline_pred = np.mean(y_train_true)
            baseline_rmse = math.sqrt(np.mean((y_test_true - baseline_pred) ** 2))
            r2 = 1.0 - (rmse ** 2) / (baseline_rmse ** 2) if baseline_rmse > 0 else float('nan')
            pred_std = np.std(test_preds)

            total_baseline_rmse += baseline_rmse
            total_r2 += r2
            total_pred_std += pred_std
            
        count += 1

    if count > 0:
        if type == "prob":
            avg_f1 = total_f1 / count
            print(f"\nAggregated Results over {count} episodes:")
            print(f"Average F1 Score: {avg_f1:.4f}")
        else:
            avg_rmse = total_rmse / count
            avg_baseline_rmse = total_baseline_rmse / count
            avg_r2 = total_r2 / count
            avg_pred_std = total_pred_std / count
            print(f"\nAggregated Results over {count} episodes:")
            print(f"Average Model RMSE:    {avg_rmse:.4f}")
            print(f"Average Baseline RMSE: {avg_baseline_rmse:.4f}")
            print(f"Average R²:            {avg_r2:.4f}")
            print(f"Average Pred Std:      {avg_pred_std:.6f}")
            if avg_pred_std < 0.01:
                print(f"\n*** VERDICT: Model has COLLAPSED (avg pred std = {avg_pred_std:.6f}) ***")
            else:
                print(f"\nVerdict: Model appears to produce varied predictions.")
    else:
        print("No episodes evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GRU on multiple episodes")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing episode CSV files")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--type", type=str, default="prob", help="Type of architecture (prob, reg, hid)")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to evaluate")
    
    args = parser.parse_args()
    
    evaluate_episodes(args.data_dir, args.model, args.type, args.num_episodes)
