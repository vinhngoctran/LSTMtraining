# Credit to Vinh Ngoc Tran (vinht@umich.edu) - University of Michigan, Ann Arbor

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import multiprocessing
from scipy.io import savemat
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
multiprocessing.set_start_method('spawn', force=True)
import numpy as np

def formdata_EXP1(file_path):
    
    data = sio.loadmat(file_path)
    MaskDat = data['MaskDat'].astype('float32')
    ClimateDat = data['ClimateDat'].astype('float32')
    TargetDat = data['TargetDat'].astype('float32')
    
    # Normalize ClimateDat and TargetDat
    scaler_climate = StandardScaler()
    scaler_target = StandardScaler()
    
    ClimateDat_normalized = scaler_climate.fit_transform(ClimateDat)
    TargetDat_normalized = scaler_target.fit_transform(TargetDat)
    
    # Get dimensions
    total_days, num_variables = ClimateDat_normalized.shape
    sequence_length = 365
    
    # Create 3D input data with lookback period of 365 days
    X = np.zeros((total_days - sequence_length + 1, sequence_length, num_variables))
    for i in range(total_days - sequence_length + 1):
        X[i] = ClimateDat_normalized[i:i+sequence_length]
    
    # Prepare target data
    Y = TargetDat_normalized[sequence_length-1:]
    
    # Prepare mask for splitting (use the last day of each sequence)
    split_mask = MaskDat[sequence_length-1:,0]
    
    # Split data using MaskDat
    Xtrain = X[split_mask == 1]
    Ytrain = Y[split_mask == 1]
    
    Xval = X[split_mask == 2]
    Yval = Y[split_mask == 2]
    
    Xtest = X[split_mask == 3]
    Ytest = Y[split_mask == 3]

    # Remove rows with NaN values from training and validation sets
    train_mask = ~np.isnan(Xtrain).any(axis=(1,2)) & ~np.isnan(Ytrain).any(axis=1)
    Xtrain = Xtrain[train_mask]
    Ytrain = Ytrain[train_mask]

    val_mask = ~np.isnan(Xval).any(axis=(1,2)) & ~np.isnan(Yval).any(axis=1)
    Xval = Xval[val_mask]
    Yval = Yval[val_mask]
    
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, scaler_climate, scaler_target


    

# Custom LSTM model
class LSTM_custom(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTM_custom, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


import os
import logging
import psutil


def get_ram_usage():
    """Get current RAM usage in GB"""
    return psutil.virtual_memory().used / (1024 * 1024 * 1024)

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    return 0

def log_memory_usage():
    ram_usage = get_ram_usage()
    gpu_usage = get_gpu_memory_usage()
    logging.info(f"RAM Usage: {ram_usage:.2f} GB, GPU Memory Usage: {gpu_usage:.2f} GB")
import gc
def clear_memory():
    # Clear Python's internal memory
    gc.collect()

    # Clear PyTorch's CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Attempt to clear system memory (Linux only)
    if os.name == 'posix':
        os.system('sync')
        

    # Print current memory usage
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")




def nse_loss(predicted, observed):
    numerator = torch.sum((observed - predicted) ** 2)
    denominator = torch.sum((observed - torch.mean(observed)) ** 2)
    return (numerator / denominator)

from scipy import stats


def compute_metrics(observed, simulated):
    """
    Compute KGE, MSE, RMSE, R2, and NSE metrics for multi-dimensional output.
    
    Parameters:
    observed (array-like): Observed values, shape (n_samples, 10)
    simulated (array-like): Simulated values, shape (n_samples, 10)
    
    Returns:
    dict: A dictionary containing the computed metrics for each dimension
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    n_dimensions = 1  # Should be 10
    metrics = {
        'nse': np.zeros(n_dimensions),
        'kge': np.zeros(n_dimensions),
        'mse': np.zeros(n_dimensions),
        'rmse': np.zeros(n_dimensions),
        'r2': np.zeros(n_dimensions)
    }
    
    for dim in range(n_dimensions):
        obs = observed[:, ]
        sim = simulated[:, ]
        
        # Remove any pairs where either observed or simulated is NaN or inf
        valid = np.isfinite(obs) & np.isfinite(sim)
        obs = obs[valid]
        sim = sim[valid]
        
        if len(obs) == 0:
            print(f"Warning: No valid data for dimension {dim}")
            continue
        
        mean_obs = np.mean(obs)
        mean_sim = np.mean(sim)
        std_obs = np.std(obs)
        std_sim = np.std(sim)
        
        # Compute correlation coefficient
        if std_obs > 0 and std_sim > 0:
            r, _ = stats.pearsonr(obs, sim)
        else:
            r = 0
        
        # Compute KGE
        kge = 1 - np.sqrt((r - 1)**2 + ((std_sim / std_obs) - 1)**2 + ((mean_sim / mean_obs) - 1)**2)
        
        # Compute MSE
        mse = np.mean((obs - sim)**2)
        
        # Compute RMSE
        rmse = np.sqrt(mse)
        
        # Compute R2
        ss_tot = np.sum((obs - mean_obs)**2)
        ss_res = np.sum((obs - sim)**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Compute NSE
        nse = 1 - (np.sum((obs - sim)**2) / np.sum((obs - mean_obs)**2))
        
        metrics['nse'][dim] = nse
        metrics['kge'][dim] = kge
        metrics['mse'][dim] = mse
        metrics['rmse'][dim] = rmse
        metrics['r2'][dim] = r2
    
    return metrics




class ResultLogger:
    def __init__(self):
        self.results = []

    def update(self, event, optimizer):
        self.results.append(optimizer.res[-1])

result_logger = ResultLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




FolderName = "EXP_4"

for idy in range(645):  # Python uses 0-based indexing, so we start from 299
    idx = 470-idy	
    filename = FolderName+f'/Results/{idx+1}.mat'
    Savemodel = FolderName+f'/Temp/{idx+1}.pth'
    Inputname =  FolderName+f'/Input/{idx+1}.mat'
    try:
        if not os.path.exists(filename):
            print(idx+1)  # Print i+1 to match MATLAB's 1-based indexing in output
            X_train, y_train, X_val,y_val, X_test, y_test, scaler_climate, scaler_target = formdata_EXP1(Inputname) 
            input_size = X_train.shape[2]
            hidden_size = 256
            num_layers = 2
            output_size = 1
            learning_rate = 0.0001
            num_epochs = 100
            batch_size = 256
            dropout_rate = 0.4
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test)  
    
    
            model = LSTM_custom(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
            #criterion = nse_loss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop with validation
            train_losses = []
            val_losses = []
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            patience = 10
            counter = 0
            
            for epoch in range(num_epochs):
                model.train()
                total_train_loss = 0
                train_batches = 0
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size].to(device)
                    batch_y = y_train[i:i+batch_size].to(device)
            
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = nse_loss(outputs, batch_y)                
                    print('Train loss: ',loss.item())
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        optimizer.step()
                        total_train_loss += loss.item()
                        train_batches += 1
                    
                avg_train_loss = total_train_loss / train_batches
                train_losses.append(avg_train_loss)
            
                try:
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        total_val_loss = 0
                        val_batches = 0
                        for i in range(0, len(X_val), batch_size):
                            batch_X = X_val[i:i+batch_size].to(device)
                            batch_y = y_val[i:i+batch_size].to(device)
                            outputs = model(batch_X)
                            loss = nse_loss(outputs, batch_y)
                            print('Val loss: ',loss.item())
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                total_val_loss += loss.item()
                                val_batches += 1
                                    
                        avg_val_loss = total_val_loss / val_batches
                        val_losses.append(avg_val_loss)
                
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        counter = 0
                        # Save the best model
                        torch.save(model.state_dict(), Savemodel)
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Early stopping")
                except:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
                    # Early stopping
                    if avg_train_loss < best_train_loss:
                        best_train_loss = avg_train_loss
                        counter = 0
                        # Save the best model
                        torch.save(model.state_dict(), Savemodel)
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Early stopping")
                      
            
            # Load the best model for testing
            model.load_state_dict(torch.load(Savemodel))
            model.eval()
            
            # Testing
            with torch.no_grad():
                y_pred = model(X_test.to(device)).cpu().numpy()
                y_true = y_test.numpy()
            
            y_pred = scaler_target.inverse_transform(y_pred)
            y_true = scaler_target.inverse_transform(y_true)
            
            # Compute metrics
            metrics = compute_metrics(y_true, y_pred)
            # Prepare data to save
            results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'train_losses': np.array(train_losses),
            'val_losses': np.array(val_losses),  # Convert back to positive loss values
        }
            savemat(filename, results)
            # Print or use the metrics
            for metric_name, values in metrics.items():
                print(f"{metric_name.upper()}:")
                for dim, value in enumerate(values):
                    print(f"  Dimension {dim}: {value:.4f}")
            
            # Plot results
            #plt.figure(figsize=(10, 5))
            #plt.plot(y_true, label='Observed')
            #plt.plot(y_pred, label='Predicted')
            #plt.legend()
            #plt.title('Observed vs Predicted')
            #plt.show()
            
            
    
    
            del model
            del optimizer
            torch.cuda.empty_cache() 
            gc.collect() 
           
            del X_train, y_train, X_val, y_val, X_test, y_test
            gc.collect() 
    except:          
            print('Error')
