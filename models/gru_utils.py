import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

# Add project root to path if needed to import constants
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.constants import LOOKBACK_WINDOW, SimulationConfig

def load_episode_data(file_path):
    """
    Loads a specific episode CSV file.
    Returns the dataframe and the numpy array of data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Expected columns: t, S, theta, kappa, sigma
    return df

def create_sequences(data, input_cols=['S'], target_col='theta', lookback=LOOKBACK_WINDOW):
    """
    Creates sequences for GRU training.
    
    Args:
        data (pd.DataFrame): The input dataframe.
        input_cols (list): List of column names to use as features.
        target_col (str): Column name to use as target.
        lookback (int): Sequence length.
        
    Returns:
        np.array: X sequences of shape (num_samples, lookback, num_features)
        np.array: y targets of shape (num_samples,)
    """
    data_values = data[input_cols].values
    target_values = data[target_col].values
    
    samples = []
    targets = []
    
    # Ensure we have enough data
    if len(data) <= lookback:
        return np.array([]), np.array([])
        
    for i in range(len(data) - lookback):
        sample = data_values[i : i + lookback]
        outcome = target_values[i + lookback]
        samples.append(sample)
        targets.append(outcome)
        
    X = np.array(samples)
    y = np.array(targets)

    if target_col == 'theta':
        # Map theta values to class indices
        # Use rounding to avoid float precision issues
        theta_map = {round(val, 1): i for i, val in enumerate(SimulationConfig.theta_values)}
        y_mapped = []
        for val in y:
            mapped_val = theta_map.get(round(float(val), 1))
            if mapped_val is None:
                # Fallback or error
                print(f"Warning: Unknown theta value {val}")
                mapped_val = 0
            y_mapped.append(mapped_val)
        y = np.array(y_mapped, dtype=int)
        
    return X, y

def prepare_dataloaders(X, y, batch_size=32, test_size=0.2, shuffle_train=True, task_type="reg"):
    """
    Splits data into train/test and creates DataLoaders.
    
    Args:
        X (np.array): Input sequences.
        y (np.array): Targets.
        batch_size (int): Batch size.
        test_size (float or int): If float, ratio of test data. If int, number of test samples.
        shuffle_train (bool): Whether to shuffle training data.
        task_type (str): "reg" (regression) or "prob" (classification).
        
    Returns:
        train_loader, test_loader, (X_train, y_train), (X_test, y_test)
    """
    num_samples = len(X)
    if isinstance(test_size, float):
        test_len = int(num_samples * test_size)
    else:
        test_len = test_size
        
    train_len = num_samples - test_len
    
    if train_len <= 0:
        raise ValueError("Test size is larger than dataset size.")
        
    X_train = X[:train_len]
    y_train = y[:train_len]
    X_test = X[train_len:]
    y_test = y[train_len:]
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    if task_type == "prob":
        # For classification, targets should be LongTensor and 1D (batch_size)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    else:
        # For regression, targets should be FloatTensor and 2D (batch_size, 1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)
