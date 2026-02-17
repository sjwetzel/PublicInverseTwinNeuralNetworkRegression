# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:23:49 2025

@author: Wetzel
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

from data_utils import num_data_points,num_anchors,ground_truth,sample_data,enforce_boundaries,perturb_data,low,high,dimX,dimY,K_MAX,noise

def mean_coord_distance(d, k, num_pts):
    Vd = math.pi**(d/2) / math.gamma(1 + d/2)
    cd = math.gamma(d/2) / (math.sqrt(math.pi) * math.gamma((d+1)/2))
    return cd * (k / (num_pts * Vd))**(1/d)

def naive_coord_distance(d,k,num_pts):#
    return (k/num_pts/10)**(1/d)

class PairGenerator(Sequence):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.steps_per_epoch = int(10*num_data_points/batch_size*K_MAX )
        self.sample_ranges = (high-low)*mean_coord_distance(dimX,2,num_anchors)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        
        X_forward_a = sample_data(self.batch_size)
        X_forward_b = perturb_data(X_forward_a, self.sample_ranges)
        
        Y_forward_a = ground_truth(X_forward_a)
        Y_forward_b = ground_truth(X_forward_b)
        # Form inverse pairs
        DY = np.concatenate([Y_forward_a, Y_forward_b, X_forward_b],axis=-1)
        DX = X_forward_a-X_forward_b
        
        return DY, DX
    
### Functions
# Split Data
def split(x, y, val_pct = 0.2, test_pct = 0.2, seed = None):
    n = len(x)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(n * test_pct)
    n_val = int(n * val_pct)
    n_train = n - n_test - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])

# Define Distance
def euclid_distance(a, b):
    return np.linalg.norm(a-b)

# Data Pairs
def make_pairs(y, x, k):
    DY, DX = [], []
    n = len(y)

    nn = NearestNeighbors(n_neighbors=k+1, metric=euclid_distance).fit(x)
    distances, indices = nn.kneighbors(x)
    for i in range(n):
        for j in indices[i]:
            DY.append(np.concatenate([y[i], y[j], x[j]]))
            DX.append(x[i] - x[j])
    return np.array(DY), np.array(DX)


# Inference
def predict_all(y_new, k, model, y_anchor, x_anchor):

    nn = NearestNeighbors(n_neighbors=k+1, metric=euclid_distance).fit(y_anchor)
    distances, indices = nn.kneighbors(y_new)
    k_indices = indices[:, k]
    
    y_kth_neighbors = y_anchor[k_indices]
    x_kth_neighbors = x_anchor[k_indices]

    DY_new = np.concatenate([y_new, y_kth_neighbors,x_kth_neighbors], axis=1)
    diff_new = model.predict(DY_new,verbose=0)

    preds = diff_new + x_kth_neighbors
    return preds

def predict_all_knn(y_new, k, y_anchor, x_anchor):

    nn = NearestNeighbors(n_neighbors=k+1, metric=euclid_distance).fit(y_anchor)
    distances, indices = nn.kneighbors(y_new)
    k_indices = indices[:, k]
    
    x_kth_neighbors = x_anchor[k_indices]
    return x_kth_neighbors


collect_ITNNR_mse_list = []
collect_kNN_mse_list = []
collect_best_ITNNR_mse = []
collect_best_ITNNR_mse_learned = []
for i in range(5):
    print(f'run {i}')
    ### Training
    np.random.seed(i)
    keras.utils.set_random_seed(i)
    
    X = sample_data(num_data_points)
    Y = ground_truth(X)
    Y = Y * np.random.normal(1.0, noise, size=Y.shape)
    # Train Val Test Split
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split(X, Y, val_pct=0.2, test_pct=0.2, seed=i)
    
    x_anchor = x_train[:num_anchors]
    y_anchor = y_train[:num_anchors]
    
    DY_train, DX_train = make_pairs(y_train, x_train, k=K_MAX)
    DY_val, DX_val = make_pairs(y_val, x_val, k=K_MAX)
    
    pair_gen = PairGenerator(batch_size=32)
    
  
    
    # forward model for selection
    forward_model = keras.Sequential([
        layers.Dense(640, activation="relu", input_shape=(x_train.shape[-1],)),
        layers.Dense(640, activation="relu"),
        layers.Dense(y_train.shape[-1])
    ])
    # 3. Compile the model
    forward_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss="mae"
    )
    # early stopping 
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=int(10000/num_data_points), # Stop if no improvement after 10 epochs
        restore_best_weights=True # Restore best weights
    )
    # learning rate decay
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=int(4000/num_data_points))
    
    # Train on data
    history = forward_model.fit(x_train,y_train,
        validation_data=(x_val, y_val),
        epochs=int(200000/num_data_points),       # Upper limit 200
        batch_size=32,
        callbacks=[early_stopper,reduce_lr], # Add callback
        verbose=0
    )
    
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend(["Train MSE", "Val MAE"])
    plt.title("Training Curve Forward Model")
    plt.show()
    
    # inverse model
    
    model = keras.Sequential([
        layers.Dense(640, activation="relu", input_shape=(DY_train.shape[-1],)),
        layers.Dense(640, activation="relu"),
        layers.Dense(X.shape[-1])
    ])
    # 3. Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss="mae"
    )
    # early stopping 
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=100, # Stop if no improvement after 10 epochs
        restore_best_weights=True # Restore best weights
    )
    # learning rate decay
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=40)
    
    # Train on data
    history = model.fit(DY_train,DX_train,
        validation_data=(DY_val, DX_val),
        epochs=2000,       # Upper limit 200
        batch_size=32,
        callbacks=[early_stopper,reduce_lr], # Add callback
        verbose=0
    )
    # Train on generator
    # history = model.fit(
    #     pair_gen,
    #     validation_data=(DY_val, DX_val),
    #     epochs=2000,       # Upper limit 200
    #     batch_size=32,
    #     callbacks=[early_stopper,reduce_lr], # Add callback
    #     verbose=0
    # )
    
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend(["Train MSE", "Val MAE"])
    plt.title("Training Curve")
    plt.show()
    
    # Store the MSE for each k
    ITNNR_mse_list = []
    kNN_mse_list = []
    
    ITNNR_predictions = []
    ITNNR_errors = []
    ITNNR_errors_learned = []
    
    k_range = range(1, K_MAX + 1)
    for k in k_range:
        print(f"k = {k}")
        
        x_preds = predict_all(y_test, k, model, y_anchor, x_anchor)
        x_preds_knn = predict_all_knn(y_test, k, y_anchor, x_anchor)
        y_reconstructed = ground_truth(enforce_boundaries(x_preds))
        y_reconstructed_learned = forward_model.predict(enforce_boundaries(x_preds),verbose=0)
        y_reconstructed_knn = ground_truth(enforce_boundaries(x_preds_knn))
    
        ITNNR_predictions.append(x_preds)
        ITNNR_errors.append((y_reconstructed - y_test)**2)
        ITNNR_errors_learned.append((y_reconstructed_learned - y_test)**2)
    
        ITNNR_mse = np.mean((y_reconstructed - y_test)**2)
        kNN_mse =  np.mean((y_reconstructed_knn - y_test)**2)
    
        print(f"  ITNNR MSE = {ITNNR_mse:.6f}")
        print(f"  knn   MSE = {kNN_mse:.6f}")
    
        ITNNR_mse_list.append(ITNNR_mse)
        kNN_mse_list.append(kNN_mse)
        
    # Stack errors 
    errors_stacked = np.stack(ITNNR_errors, axis=0)
    errors_learned_stacked = np.stack(ITNNR_errors_learned, axis=0)
    
    # pick which anchor k has smallest error
    best_idx = np.argmin(errors_stacked, axis=0)
    best_idx_learned = np.argmin(errors_learned_stacked, axis=0)
    
    # Stack predictions
    combined_predictions = np.stack(ITNNR_predictions, axis=0)
    
    # Select elementwise using fancy indexing
    best_predictions = combined_predictions[best_idx, np.arange(combined_predictions.shape[1])[:,None], np.arange(combined_predictions.shape[2])]
    best_predictions_learned = combined_predictions[best_idx_learned, np.arange(combined_predictions.shape[1])[:,None], np.arange(combined_predictions.shape[2])]
    
    best_y_reconstructed = ground_truth(enforce_boundaries(best_predictions))
    best_ITNNR_mse = np.mean((best_y_reconstructed - y_test)**2)
    
    print(f"\n  best ITNNR MSE = {best_ITNNR_mse:.6f}")
    
    
    best_y_reconstructed_learned = ground_truth(enforce_boundaries(best_predictions_learned))
    best_ITNNR_mse_learned = np.mean((best_y_reconstructed_learned - y_test)**2)
    
    print(f"\n  best ITNNR MSE learned = {best_ITNNR_mse_learned:.6f}")
    
    # Plots
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, ITNNR_mse_list, marker="o", linestyle='-', color='b')
    plt.xlabel("k (Number of Neighbors)", fontsize=12)
    plt.ylabel("Best Inverse Model MSE (in y_p space)", fontsize=12)
    plt.title("Best MSE vs. k", fontsize=14)
    plt.xticks(k_range)
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.show()
    
    collect_ITNNR_mse_list.append(ITNNR_mse_list)
    collect_kNN_mse_list.append(kNN_mse_list)
    collect_best_ITNNR_mse.append(best_ITNNR_mse)
    collect_best_ITNNR_mse_learned.append(best_ITNNR_mse_learned)    

np.savez("ITNNR_noisy_pair_data_results.npz", ITNNR_mse_list=np.array(collect_ITNNR_mse_list), kNN_mse_list=np.array(collect_kNN_mse_list),best_ITNNR_mse=np.array(collect_best_ITNNR_mse),best_ITNNR_mse_learned=np.array(collect_best_ITNNR_mse_learned))
