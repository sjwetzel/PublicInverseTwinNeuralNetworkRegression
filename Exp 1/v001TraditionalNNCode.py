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

collect_NN_mse = []
X = sample_data(num_data_points)
Y = ground_truth(X)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = split(X, Y, val_pct=0.2, test_pct=0.2, seed=0)

for i in range(5):

    ### Training
    np.random.seed(i)
    keras.utils.set_random_seed(i)
    
    
    model = keras.Sequential([
        layers.Dense(640, activation="relu", input_shape=(y_train.shape[-1],)),
        layers.Dense(640, activation="relu"),
        layers.Dense(x_train.shape[-1])
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
    history = model.fit(y_train,x_train,
        validation_data=(y_val, x_val),
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
    
    x_preds = model.predict(y_test)
    y_reconstructed = ground_truth(enforce_boundaries(x_preds))
    
    NN_mse = np.mean((y_reconstructed - y_test)**2)
    
    print(f"  NN MSE = {NN_mse:.6f}")

    collect_NN_mse.append(NN_mse)

np.savez("traditional_NN_results.npz", NN_mse=np.array(collect_NN_mse))
